import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from typing import Tuple, Dict
from datetime import datetime, timedelta
import logging

# -------- File Paths --------
import os
_d = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MEETING_ROOMS_CSV, WORKSTATIONS_CSV, BOOKINGS_CSV = [os.path.join(_d, f) for f in ["meeting_rooms.csv", "workstations.csv", "bookings.csv"]]

# ---------------- Logging Configuration ---------------- #

def setup_room_selection_logger(enable_logging: bool = False, log_level: str = "INFO") -> logging.Logger:
    """
    Configure logger for room selection reasoning
    
    Args:
        enable_logging: Whether to enable console logging
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("room_selector")
    logger.handlers.clear()  # Clear existing handlers
    
    if enable_logging:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.setLevel(getattr(logging, log_level.upper()))
        logger.disabled = False
    else:
        logger.disabled = True
    
    return logger

# Initialize logger (can be reconfigured later)
room_logger = setup_room_selection_logger()

# ---------------- Utility Functions ---------------- #

def ensure_string_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna('')
    return df

@st.cache_data(show_spinner=False)
def load_rooms(path: str) -> pd.DataFrame:
    """Load and preprocess rooms data with caching and error handling."""
    try:
        df = pd.read_csv(path)
        df = compute_room_centers(normalize_columns(df))
        return df
    except Exception as e:
        st.error(f"Failed to load meeting rooms data: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_workstations(path: str) -> pd.DataFrame:
    """Load and preprocess workstations data with caching and error handling."""
    try:
        df = pd.read_csv(path)
        return normalize_columns(df)
    except Exception as e:
        st.error(f"Failed to load workstations data: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_bookings(path: str) -> pd.DataFrame:
    """Load and preprocess bookings data with caching and error handling."""
    try:
        df = pd.read_csv(path)
        df = preprocess_bookings_raw(normalize_columns(df))
        df = ensure_string_columns(df, ['workstation_id','room_id','start_time','end_time','date','booking_id'])
        return df
    except Exception as e:
        st.error(f"Failed to load bookings data: {e}")
        return pd.DataFrame()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def preprocess_bookings_raw(bookings: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and sanitize raw bookings:
    - Standardize boolean columns
    - Drop duplicate booking_id (keep earliest start_dt)
    - Remove impossible time windows (missing/invalid times, non-positive duration, absurd durations)
    - Return cleaned DataFrame with start_dt and end_dt columns
    """
    b = bookings.copy()

    # Normalize column names
    b.columns = [c.strip().lower() for c in b.columns]

    # Normalize booleans robustly (avoid FutureWarning on fillna downcasting)
    bool_cols = ["is_recurring", "has_catering", "has_av_equipment", "cancelled", "no_show"]
    for c in bool_cols:
        if c in b.columns:
            # Convert any representation to standardized boolean
            raw = b[c]
            # Treat explicit True/False strings, 1/0, and actual bools
            upper = raw.astype(str).str.strip().str.upper()
            mapped = upper.map({"TRUE": True, "FALSE": False, "1": True, "0": False})
            # Where mapping failed, default to False
            b[c] = mapped.fillna(False).astype(bool)

    # Coerce duration to numeric and filter invalid
    if "duration_minutes" in b.columns:
        b["duration_minutes"] = pd.to_numeric(b["duration_minutes"], errors="coerce")
    else:
        b["duration_minutes"] = np.nan

    # Robust time parsing
    # Accept HH:MM; drop rows that fail parsing
    if "date" not in b.columns:
        raise ValueError("bookings data missing 'date' column")
    if "start_time" not in b.columns:
        raise ValueError("bookings data missing 'start_time' column")

    # Combine datetime; coerce errors to NaT
    b["start_dt"] = pd.to_datetime(b["date"].astype(str) + " " + b["start_time"].astype(str), errors="coerce")
    # If duration exists, compute end_dt
    b["end_dt"] = b["start_dt"] + pd.to_timedelta(b["duration_minutes"], unit="m", errors="coerce")

    # Drop rows with missing criticals
    b = b.dropna(subset=["booking_id", "room_id", "start_dt", "duration_minutes"]).copy()

    # Remove impossible durations: <= 0 or absurdly large (e.g., > 12 hours)
    b = b[(b["duration_minutes"] > 0) & (b["duration_minutes"] <= 12 * 60)].copy()

    # Recompute end_dt after filtering
    b["end_dt"] = b["start_dt"] + pd.to_timedelta(b["duration_minutes"], unit="m")

    # Optional: constrain date range to a sensible window (e.g., +/- 5 years from now)
    now = pd.Timestamp.now(tz=None).normalize()
    min_date = now - pd.Timedelta(days=5*365)
    max_date = now + pd.Timedelta(days=5*365)
    b = b[(b["start_dt"] >= min_date) & (b["start_dt"] <= max_date)].copy()

    # Deduplicate on booking_id: keep the earliest consistent start_dt
    b = b.sort_values(["booking_id", "start_dt", "duration_minutes"], ascending=[True, True, True])
    b = b.drop_duplicates(subset=["booking_id"], keep="first").reset_index(drop=True)

    # Ensure numeric fields
    for c in ["booked_attendees", "actual_attendees", "booking_lead_time_hours"]:
        if c in b.columns:
            b[c] = pd.to_numeric(b[c], errors="coerce")

    # Fill NaNs in expected numeric columns with zeros where safe for modeling
    b["booked_attendees"] = b["booked_attendees"].fillna(0)
    b["actual_attendees"] = b["actual_attendees"].fillna(0)
    b["booking_lead_time_hours"] = b["booking_lead_time_hours"].fillna(0)

    # Standardize strings
    for c in ["purpose", "department"]:
        if c in b.columns:
            b[c] = b[c].fillna("Unknown").astype(str)

    # Keep original date and start_time strings for UI/compat if needed
    # Also regenerate canonical date/start_time strings from start_dt to ensure consistency
    b["date"] = b["start_dt"].dt.strftime("%Y-%m-%d")
    b["start_time"] = b["start_dt"].dt.strftime("%H:%M")

    # Recalculate end time fields if you need them for debug (not required for your selectors)
    b["end_time"] = b["end_dt"].dt.strftime("%H:%M")

    return b

def compute_room_centers(rooms: pd.DataFrame) -> pd.DataFrame:
    rooms = rooms.copy()
    rooms["center_x_calc"] = rooms["x"] + rooms["width"] / 2.0
    rooms["center_y_calc"] = rooms["y"] + rooms["height"] / 2.0
    rooms["right"] = rooms["x"] + rooms["width"]
    rooms["top"] = rooms["y"] + rooms["height"]
    return rooms

def is_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a

def get_meeting_window(date: str, start_time: str, duration_minutes: int) -> Tuple[str, int, int]:
    h, m = map(int, start_time.split(":"))
    start_min = h * 60 + m
    end_min = start_min + duration_minutes
    return date, start_min, end_min

def filter_capacity_tier(rooms: pd.DataFrame, required_capacity: int) -> Tuple[pd.DataFrame, str]:
    exact = rooms[rooms["capacity"] == required_capacity]
    if not exact.empty:
        room_logger.info(f"Found {len(exact)} rooms with exact capacity match ({required_capacity}): {exact['id'].tolist()}")
        return exact, "Exact capacity tier applied"
    
    over = rooms[rooms["capacity"] > required_capacity]
    if over.empty:
        room_logger.warning(f"No rooms meet capacity requirement of {required_capacity}. Max available: {rooms['capacity'].max() if not rooms.empty else 'N/A'}")
        return pd.DataFrame(), "No rooms meet capacity requirement"
    
    min_over = over["capacity"].min()
    min_over_rooms = over[over["capacity"] == min_over]
    room_logger.info(f"  No exact match found. Selected {len(min_over_rooms)} rooms with minimal over-capacity ({min_over}): {min_over_rooms['id'].tolist()}")
    room_logger.debug(f"   Rejected rooms with higher capacity: {over[over['capacity'] > min_over]['id'].tolist()}")
    
    return min_over_rooms, f"Minimal over-capacity tier applied (capacity={min_over})"

def filter_available_optimized(rooms: pd.DataFrame, bookings: pd.DataFrame, meeting_date: str, start_min: int, end_min: int) -> pd.DataFrame:
    """
    Optimized version using vectorized operations instead of nested loops
    """
    room_logger.info(f"Checking availability for {len(rooms)} rooms on {meeting_date} from {start_min//60:02d}:{start_min%60:02d} to {end_min//60:02d}:{end_min%60:02d}")
    
    # Filter bookings for the specific date and non-cancelled
    day_bookings = bookings[
        (bookings["date"] == meeting_date) & 
        (~bookings["cancelled"])
    ].copy()
    
    if day_bookings.empty:
        room_logger.info(f"No active bookings found for {meeting_date}. All {len(rooms)} rooms are available.")
        return rooms
    
    room_logger.debug(f"   Found {len(day_bookings)} active bookings for {meeting_date}")
    
    # Convert booking times to minutes for vectorized comparison
    day_bookings["start_minutes"] = (
        day_bookings["start_time"].str.split(":").str[0].astype(int) * 60 +
        day_bookings["start_time"].str.split(":").str[1].astype(int)
    )
    day_bookings["end_minutes"] = day_bookings["start_minutes"] + day_bookings["duration_minutes"]
    
    # Find conflicts using vectorized operations
    conflicts = day_bookings[
        (day_bookings["start_minutes"] < end_min) & 
        (day_bookings["end_minutes"] > start_min)
    ]
    
    # Get room IDs with conflicts
    conflicted_rooms = conflicts["room_id"].unique()
    available_rooms = rooms[~rooms["id"].isin(conflicted_rooms)]
    
    room_logger.info(f"Availability check complete: {len(available_rooms)}/{len(rooms)} rooms available")
    
    if len(conflicted_rooms) > 0:
        room_logger.info(f"Rooms with conflicts: {conflicted_rooms.tolist()}")
        for room_id in conflicted_rooms[:5]:  # Log details for first 5 conflicts
            room_conflicts = conflicts[conflicts["room_id"] == room_id]
            conflict_times = [f"{row['start_time']}-{row['end_time']}" for _, row in room_conflicts.iterrows()]
            room_logger.debug(f"   {room_id}: conflicts at {conflict_times}")
    
    if len(available_rooms) > 0:
        room_logger.info(f"Available rooms: {available_rooms['id'].tolist()}")
    
    return available_rooms

def compute_distances(rooms: pd.DataFrame, ws: pd.DataFrame, workstation_id: str) -> pd.DataFrame:
    ws_row = ws[ws["workstation_id"].str.upper() == workstation_id.upper()]
    if ws_row.empty:
        raise ValueError(f"Workstation {workstation_id} not found")
    
    ws_x, ws_y = ws_row.iloc[0]["x"], ws_row.iloc[0]["y"]
    room_logger.info(f"Workstation {workstation_id} located at coordinates ({ws_x}, {ws_y})")
    
    rooms = rooms.copy()
    rooms["distance_m"] = ((rooms["center_x_calc"] - ws_x)**2 + (rooms["center_y_calc"] - ws_y)**2).apply(math.sqrt)
    
    # Log distance calculations
    distances_info = rooms[["id", "name", "distance_m"]].sort_values("distance_m")
    room_logger.info(f"Distance calculations completed for {len(rooms)} rooms:")
    for _, row in distances_info.head(5).iterrows():  # Log top 5 closest
        room_logger.debug(f"   {row['id']} ({row['name']}): {row['distance_m']:.1f}m")
    
    return rooms

def compute_daily_utilization_flexible(rooms: pd.DataFrame, bookings: pd.DataFrame, meeting_date: str, 
                                       business_hours_start: int = 8, business_hours_end: int = 18) -> pd.DataFrame:
    """
    Improved utilization calculation with configurable business hours
    """
    rooms = rooms.copy()
    business_minutes = (business_hours_end - business_hours_start) * 60
    utilization_map = {}
    
    room_logger.info(f"Computing utilization for {meeting_date} (business hours: {business_hours_start:02d}:00-{business_hours_end:02d}:00)")
    
    for _, room in rooms.iterrows():
        room_bookings = bookings[
            (bookings["room_id"] == room["id"]) & 
            (bookings["date"] == meeting_date) & 
            (~bookings["cancelled"])
        ]
        total_minutes = room_bookings["duration_minutes"].sum()
        utilization = min(total_minutes / business_minutes, 1.0)  # Cap at 100%
        utilization_map[room["id"]] = utilization
        
        room_logger.debug(f"   {room['id']} ({room['name']}): {total_minutes}min booked = {utilization:.1%} utilization")
    
    rooms["utilization"] = rooms["id"].map(utilization_map).fillna(0.0)
      # Log utilization summary
    high_util_rooms = rooms[rooms["utilization"] > 0.7]
    if len(high_util_rooms) > 0:
        high_util_info = [(row['id'], f"{row['utilization']:.1%}") for _, row in high_util_rooms.iterrows()]
        room_logger.info(f" High utilization rooms (>70%): {high_util_info}")
    
    return rooms

def select_room_strict(
    rooms: pd.DataFrame,
    workstations: pd.DataFrame,
    bookings: pd.DataFrame,
    required_capacity: int,
    workstation_id: str,
    meeting_date: str,
    start_time: str,
    duration_minutes: int,
    utilization_threshold: float = 0.65,
    distance_tolerance_m: float = 5.0,
    enable_logging: bool = False,
    log_level: str = "INFO"
) -> Dict:
    # Configure logger for this session (now respects log_level)
    global room_logger
    room_logger = setup_room_selection_logger(enable_logging, log_level)
    
    room_logger.info(f"Starting room selection process:")
    room_logger.info(f"   Meeting: {meeting_date} at {start_time} for {duration_minutes}min")
    room_logger.info(f"   Capacity needed: {required_capacity}")
    room_logger.info(f"   From workstation: {workstation_id}")
    room_logger.info(f"   Utilization threshold: {utilization_threshold:.1%}, Distance tolerance: {distance_tolerance_m}m")
    
    rooms = compute_room_centers(normalize_columns(rooms))
    workstations = normalize_columns(workstations)
    bookings = normalize_columns(bookings)

    meeting_date, start_min, end_min = get_meeting_window(meeting_date, start_time, duration_minutes)

    capacity_filtered, capacity_explanation = filter_capacity_tier(rooms, required_capacity)
    if capacity_filtered.empty:
        room_logger.error("No rooms found meeting capacity requirements")
        return {"selected_room": None, "decision_explanation": capacity_explanation}

    available_rooms = filter_available_optimized(capacity_filtered, bookings, meeting_date, start_min, end_min)
    if available_rooms.empty:
        room_logger.error("All capacity-suitable rooms are booked")
        return {"selected_room": None, "decision_explanation": "All eligible rooms unavailable"}

    available_rooms = compute_distances(available_rooms, workstations, workstation_id)
    available_rooms = compute_daily_utilization_flexible(available_rooms, bookings, meeting_date)

    # Log all candidate rooms before selection
    room_logger.info(f"Final candidate rooms ({len(available_rooms)}):")
    candidates_df = available_rooms[["id", "name", "distance_m", "utilization", "capacity"]].sort_values("distance_m")
    for _, row in candidates_df.iterrows():
        room_logger.info(f"   {row['id']} ({row['name']}): {row['distance_m']:.1f}m, {row['utilization']:.1%} util, cap={row['capacity']}")

    available_rooms = available_rooms.sort_values(by=["distance_m", "utilization", "id"])
    nearest_room = available_rooms.iloc[0]
    d_min = nearest_room["distance_m"]
    
    room_logger.info(f"Nearest room: {nearest_room['id']} at {d_min:.1f}m with {nearest_room['utilization']:.1%} utilization")

    explanation = [capacity_explanation]
    if nearest_room["utilization"] > utilization_threshold:
        room_logger.warning(f"Nearest room {nearest_room['id']} utilization ({nearest_room['utilization']:.1%}) > threshold ({utilization_threshold:.1%})")
        explanation.append(f"Nearest room utilization {nearest_room['utilization']:.2f} > threshold")
        
        candidates = available_rooms[(available_rooms["distance_m"] <= d_min + distance_tolerance_m) &
                                     (available_rooms["utilization"] < nearest_room["utilization"])]
        
        room_logger.info(f"Searching for alternatives within {distance_tolerance_m}m tolerance...")
        room_logger.debug(f"   Tolerance range: {d_min:.1f}m to {d_min + distance_tolerance_m:.1f}m")
        
        if not candidates.empty:
            candidates = candidates.sort_values(by=["utilization", "distance_m", "id"])
            selected = candidates.iloc[0]
            room_logger.info(f"Found better alternative: {selected['id']} ({selected['utilization']:.1%} util, {selected['distance_m']:.1f}m)")
            
            # Log why other candidates were rejected
            rejected = candidates[candidates.index != selected.name]
            for _, room in rejected.iterrows():
                room_logger.debug(f"   Rejected {room['id']}: higher utilization ({room['utilization']:.1%}) or distance ({room['distance_m']:.1f}m)")
            
            explanation.append("Tolerance triggered: selected lower-utilization alternative")
        else:
            selected = nearest_room
            room_logger.info(f"No better alternatives found within tolerance - keeping nearest room {nearest_room['id']}")
            
            # Log why nearby rooms were rejected
            nearby_rejected = available_rooms[
                (available_rooms["distance_m"] <= d_min + distance_tolerance_m) & 
                (available_rooms.index != nearest_room.name)
            ]
            for _, room in nearby_rejected.iterrows():
                room_logger.debug(f"   Rejected nearby {room['id']}: higher utilization ({room['utilization']:.1%})")
            
            explanation.append("No better alternative found within tolerance")
    else:
        selected = nearest_room
        room_logger.info(f"Selected nearest room {nearest_room['id']} - under utilization threshold")
        explanation.append("Nearest room under utilization threshold")
    
    room_logger.info(f"FINAL SELECTION: {selected['id']} ({selected['name']})")
    room_logger.info(f"   Distance: {selected['distance_m']:.1f}m")
    room_logger.info(f"   Utilization: {selected['utilization']:.1%}")
    room_logger.info(f"   Capacity: {selected['capacity']}")

    return {
        "selected_room": {
            "id": selected["id"],
            "name": selected["name"],
            "capacity": int(selected["capacity"]),
            "distance_m": round(selected["distance_m"], 2),
            "utilization_pct": round(selected["utilization"] * 100, 1),
            "available": True
        },
        "decision_explanation": "; ".join(explanation)
    }

def validate_inputs(rooms: pd.DataFrame, workstations: pd.DataFrame, workstation_id: str, 
                   required_capacity: int, meeting_date: str, start_time: str, duration_minutes: int) -> Dict:
    """
    Comprehensive input validation with detailed error messages
    """
    errors = []
    
    # Validate dataframes
    if rooms.empty:
        errors.append("No meeting rooms available in the system")
    if workstations.empty:
        errors.append("No workstations defined in the system")
    
    # Validate workstation exists
    if not workstations.empty:
        ws_exists = workstations["workstation_id"].str.upper().eq(workstation_id.upper()).any()
        if not ws_exists:
            available_ws = ', '.join(workstations["workstation_id"].head(5).tolist())
            errors.append(f"Workstation {workstation_id} not found. Available: {available_ws}...")
    
    # Validate capacity
    if not rooms.empty:
        max_capacity = rooms["capacity"].max()
        if required_capacity > max_capacity:
            errors.append(f"Required capacity ({required_capacity}) exceeds maximum available ({max_capacity})")
    
    # Validate time format
    try:
        datetime.strptime(start_time, "%H:%M")
    except ValueError:
        errors.append(f"Invalid time format: {start_time}. Expected HH:MM")
    
    # Validate duration
    if duration_minutes <= 0:
        errors.append("Duration must be positive")
    if duration_minutes > 24 * 60:  # More than 24 hours
        errors.append("Duration exceeds 24 hours")
    
    return {"valid": len(errors) == 0, "errors": errors}

def extract_ml_features(bookings: pd.DataFrame, rooms: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features for ML models from booking history
    """
    features = bookings.copy()
    
    # Time-based features
    features["hour"] = pd.to_datetime(features["start_time"], format="%H:%M").dt.hour
    features["day_of_week"] = pd.to_datetime(features["date"]).dt.dayofweek
    features["is_weekend"] = features["day_of_week"].isin([5, 6])
    
    # Booking pattern features
    features["booking_lead_time_days"] = features["booking_lead_time_hours"] / 24
    features["attendance_ratio"] = features["actual_attendees"] / features["booked_attendees"].replace(0, 1)
    
    # Room utilization features
    features["capacity_utilization"] = features.apply(
        lambda row: row["booked_attendees"] / rooms[rooms["id"] == row["room_id"]]["capacity"].iloc[0]
        if len(rooms[rooms["id"] == row["room_id"]]) > 0 else 0, axis=1
    )
    
    return features

def calculate_room_score(room: pd.Series, workstation_pos: Tuple[float, float], 
                        weights: Dict[str, float] = None) -> float:
    """
    Multi-criteria scoring function for room selection
    
    Args:
        room: Room data series
        workstation_pos: (x, y) coordinates of workstation
        weights: Dictionary of scoring weights
    
    Returns:
        Composite score (lower is better)
    """
    if weights is None:
        weights = {
            "distance": 0.4,
            "utilization": 0.3,
            "capacity_efficiency": 0.2,
            "convenience": 0.1
        }
    
    # Distance score (normalized)
    distance = math.sqrt((room["center_x_calc"] - workstation_pos[0])**2 + 
                        (room["center_y_calc"] - workstation_pos[1])**2)
    distance_score = distance / 100.0  # Normalize to 0-1 range
    
    # Utilization score (prefer medium utilization, not too high or too low)
    utilization = room["utilization"]
    utilization_score = abs(utilization - 0.5) * 2  # Optimal around 50%
    
    # Capacity efficiency score (prefer minimal over-capacity)
    # Assuming this is calculated elsewhere and stored in room data
    capacity_score = room.get("capacity_excess", 0) / 10.0
    
    # Convenience score (could include features like AV equipment, etc.)
    convenience_score = 0.0  # Placeholder for future features
    
    # Weighted composite score
    total_score = (
        weights["distance"] * distance_score +
        weights["utilization"] * utilization_score +
        weights["capacity_efficiency"] * capacity_score +
        weights["convenience"] * convenience_score
    )
    
    return total_score

# ---------------- Floor Plan Visualization ---------------- #

def plot_floorplan(df_rooms, df_workstations, highlight_ws_id=None, highlight_room=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(df_rooms["capacity"].min(), df_rooms["capacity"].max())

    for _, row in df_rooms.iterrows():
        edgecolor = "red" if highlight_room is not None and row["id"] == highlight_room["id"] else "black"
        rect = Rectangle((row["x"], row["y"]), row["width"], row["height"], linewidth=2, edgecolor=edgecolor,
                         facecolor=cmap(norm(row["capacity"])), alpha=0.9)
        ax.add_patch(rect)
        ax.text(row["center_x_calc"], row["center_y_calc"], f"{row['name']} ({row['capacity']})",
                ha="center", va="center", color="white", fontsize=8, weight="bold")

    for _, ws in df_workstations.iterrows():
        ax.add_patch(
            Rectangle((ws["x"] - 0.5, ws["y"] - 0.5), 1, 1, linewidth=1, edgecolor="gray", facecolor="lightgreen"))
        ax.text(ws["x"], ws["y"], ws["workstation_id"], fontsize=5, ha="center", va="center")

    if highlight_ws_id:
        ws_row = df_workstations[df_workstations["workstation_id"].str.upper() == highlight_ws_id.upper()]
        if not ws_row.empty:
            px, py = ws_row.iloc[0]["x"], ws_row.iloc[0]["y"]
            ax.add_patch(Rectangle((px - 1, py - 1), 2, 2, linewidth=2, edgecolor="blue", facecolor="none"))
            ax.text(px, py + 1.5, f"Selected: {highlight_ws_id}", color="blue", fontsize=9, weight="bold", ha="center")

    if highlight_room is not None:
        cx, cy = highlight_room["center_x_calc"], highlight_room["center_y_calc"]
        ring = Circle((cx, cy), radius=max(highlight_room["width"], highlight_room["height"]) * 0.4,
                      fill=False, color="red", linewidth=2)
        ax.add_patch(ring)

    ax.set_aspect("equal")
    ax.set_xlim(0, df_rooms["right"].max() + 5)
    ax.set_ylim(0, df_rooms["top"].max() + 5)
    ax.set_title("Meeting Rooms – Floor Plan")
    st.pyplot(fig)

# ---------------- Streamlit UI ---------------- #

st.title("🏢 Smart Meeting Room Selector ")
# Prevent duplicate booking actions rendering in same run
if 'from_find_room_click' not in st.session_state:
    st.session_state.from_find_room_click = False
elif st.session_state.from_find_room_click:
    # Reset flag at start of a new rerun so persistent section can appear
    st.session_state.from_find_room_click = False

def generate_time_options():
    times = []
    for hour in range(24):
        for minute in [0, 30]:
            times.append(f"{hour:02d}:{minute:02d}")
    return times

def get_default_time_index():
    now = datetime.now()
    current_minutes = now.hour * 60 + now.minute
    
    next_slot_minutes = ((current_minutes // 30) + 1) * 30
    if next_slot_minutes >= 24 * 60:  
        next_slot_minutes = 0
    
    return next_slot_minutes // 30

def get_end_time_options(start_time: str, time_options: list) -> list:
    """Get end time options that are after the start time"""
    start_index = time_options.index(start_time)
    end_options = time_options[start_index + 1:]
    
    if len(end_options) < 2:
        next_day_options = [f"{hour:02d}:{minute:02d}" for hour in range(0, 6) for minute in [0, 30]]
        end_options.extend(next_day_options)
    
    return end_options

def get_default_end_time(start_time: str, time_options: list) -> str:
    """Get default end time (30 minutes after start time)"""
    start_index = time_options.index(start_time)
    
    if start_index + 1 < len(time_options):
        return time_options[start_index + 1]
    else:
        return "00:00"

def calculate_duration(start_time: str, end_time: str) -> tuple:
    start_h, start_m = map(int, start_time.split(":"))
    end_h, end_m = map(int, end_time.split(":"))
    
    start_minutes = start_h * 60 + start_m
    end_minutes = end_h * 60 + end_m
    
    if end_minutes <= start_minutes:
        end_minutes += 24 * 60
    
    duration_minutes = end_minutes - start_minutes
    duration_hours = duration_minutes / 60
    
    return duration_minutes, duration_hours

def get_next_weekday():
    """Get the next available weekday (Monday-Friday), starting from today"""
    today = datetime.now().date()
    
    # If today is a weekday (Monday=0, Sunday=6), return today
    # If today is weekend, return next Monday
    if today.weekday() < 5:  # Monday to Friday (0-4)
        return today
    else:
        # Calculate days to add to get to next Monday
        days_to_add = 7 - today.weekday()  # This will give us Monday
        return today + timedelta(days=days_to_add)

def is_weekday(date):
    """Check if a date is a weekday (Monday-Friday)"""
    return date.weekday() < 5

def get_weekday_range(start_date=None, days_ahead=30):
    """
    Generate a list of weekdays for the next specified number of days
    This could be used for a more restrictive date picker if needed
    """
    if start_date is None:
        start_date = datetime.now().date()
    
    weekdays = []
    current_date = start_date
    
    for _ in range(days_ahead * 2):  # Check more days to get enough weekdays
        if current_date.weekday() < 5:  # Monday to Friday
            weekdays.append(current_date)
        current_date += timedelta(days=1)
        if len(weekdays) >= days_ahead:
            break
    
    return weekdays

time_options = generate_time_options()
default_start_index = get_default_time_index()

def get_workstation_options():
    """Generate list of workstation IDs from WS-01 to WS-60"""
    return [f"WS-{i:02d}" for i in range(1, 61)]

workstation_options = get_workstation_options()

# Add logging control in sidebar
with st.sidebar:
    st.header("🔧 Settings")
    enable_detailed_logging = st.checkbox(
        "Enable Detailed Logging", 
        value=False,
        help="Show detailed reasoning for room selection decisions in console/logs"
    )
    if enable_detailed_logging:
        log_level = st.selectbox(
            "Log Level",
            options=["DEBUG", "INFO", "WARNING", "ERROR"],
            index=1,
            help="Choose the level of detail for logging output"
        )
    else:
        log_level = "INFO"
    utilization_threshold = st.slider(
        "Utilization Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
        help="Max utilization allowed for nearest room before searching alternatives"
    )
    distance_tolerance_m = st.number_input(
        "Distance Tolerance (m)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5,
        help="Extra distance tolerance when looking for lower-utilization alternatives"
    )

col1, col2 = st.columns([2, 1])

with col1:
    workstation_id = st.selectbox(
        "Select Your Workstation", 
        options=workstation_options,
        index=0,
        help="Choose your workstation from the dropdown list"
    )

with col2:
    # Show selected workstation info
    st.metric("Selected Workstation", workstation_id)

required_capacity = st.number_input("Required Capacity", min_value=1, max_value=10, value=4, help="Number of people for the meeting")

# Meeting date selection with weekend exclusion
default_date = get_next_weekday()

# Create a custom date input with weekend validation
col_date, col_info = st.columns([2, 1])

with col_date:
    selected_date = st.date_input(
        "Meeting Date", 
        value=default_date,
        help="Select a weekday (Monday-Friday). Weekends are not allowed."
    )

with col_info:
    # Real-time feedback about selected date
    if selected_date.weekday() >= 5:  # Saturday=5, Sunday=6
        st.write("<small style='color: red;'>❌ Weekend</small>", unsafe_allow_html=True)
        meeting_date = None
    else:
        day_name = selected_date.strftime("%A")
        st.write(f"<small style='color: green;'>✓ {day_name}</small>", unsafe_allow_html=True)
        meeting_date = selected_date.strftime("%Y-%m-%d")

# Show weekend warning if weekend is selected
if selected_date.weekday() >= 5:
    st.warning("Weekend dates are not allowed for meeting bookings. Please select a weekday (Monday-Friday).")
    
    # Suggest next weekday
    next_weekday = selected_date
    while next_weekday.weekday() >= 5:
        next_weekday = next_weekday + timedelta(days=1)
    st.info(f"Next available weekday: {next_weekday.strftime('%A, %B %d, %Y')}")
else:
    # Show selected weekday confirmation
    st.info(f"Selected meeting date: {selected_date.strftime('%A, %B %d, %Y')}")

col1, col2 = st.columns(2)
with col1:
    start_time = st.select_slider("Start Time", options=time_options, value=time_options[default_start_index])

end_time_options = get_end_time_options(start_time, time_options)

if 'previous_start_time' not in st.session_state:
    st.session_state.previous_start_time = start_time

if st.session_state.previous_start_time != start_time:
    st.session_state.previous_start_time = start_time
    default_end_time = get_default_end_time(start_time, time_options)
    if 'end_time_key' not in st.session_state:
        st.session_state.end_time_key = 0
    st.session_state.end_time_key += 1

with col2:
    end_time_options = get_end_time_options(start_time, time_options)
    
    if len(end_time_options) >= 2:  # Ensure we have enough options for slider
        default_end_time = get_default_end_time(start_time, time_options)
        if default_end_time not in end_time_options:
            default_end_time = end_time_options[0]
        
        end_time = st.select_slider(
            "End Time", 
            options=end_time_options, 
            value=default_end_time,
            key=f"end_time_{st.session_state.get('end_time_key', 0)}"
        )
    elif len(end_time_options) == 1:
        # If only one option, use selectbox instead of slider
        end_time = st.selectbox(
            "End Time",
            options=end_time_options,
            index=0,
            key=f"end_time_select_{st.session_state.get('end_time_key', 0)}"
        )
    else:
        st.warning("No valid end times available")
        end_time = start_time

if start_time and end_time and end_time != start_time:
    duration_minutes, duration_hours = calculate_duration(start_time, end_time)
    if duration_minutes > 0:
        start_h = int(start_time.split(":")[0])
        end_h = int(end_time.split(":")[0])
        next_day_text = " (next day)" if end_h < start_h else ""
        
        if duration_hours.is_integer():
            st.info(f"**Total booking duration:** {int(duration_hours)} hour(s){next_day_text}")
        else:
            st.info(f"**Total booking duration:** {duration_hours:.1f} hours ({duration_minutes} minutes){next_day_text}")
    else:
        st.warning("End time must be after start time!")
else:
    st.warning("Please select valid start and end times!")

import uuid
# NOTE: portalocker optional; add to requirements.txt for robust cross-process locking.
try:
    import portalocker  # type: ignore
except ImportError:
    portalocker = None  # Fallback if not installed yet

BOOKING_LOCK_FILE = os.path.join(_d, "bookings.lock")

def _acquire_lock(path: str):
    """Acquire an exclusive lock. Uses portalocker if available; otherwise manual file lock."""
    if portalocker:
        return portalocker.Lock(path, timeout=5)
    # Fallback simple context manager
    class _BasicLock:
        def __enter__(self_inner):
            # Busy wait up to 5 seconds
            import time
            start = time.time()
            while os.path.exists(path) and time.time() - start < 5:
                time.sleep(0.1)
            # Create lock marker
            with open(path, 'w') as f:  # create marker file
                f.write(str(datetime.utcnow()))
            return self_inner
        def __exit__(self_inner, exc_type, exc, tb):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
    return _BasicLock()

def _compute_end_time(start_time: str, duration_minutes: int) -> str:
    h, m = map(int, start_time.split(':'))
    total = h*60 + m + duration_minutes
    total %= (24*60)
    return f"{total//60:02d}:{total%60:02d}"

def _time_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end

def workstation_has_conflict(bookings: pd.DataFrame, workstation_id: str, meeting_date: str, start_time: str, duration_minutes: int) -> Tuple[bool, pd.DataFrame]:
    """Return True and conflicting rows if workstation already booked for overlapping slot."""
    if bookings.empty or 'workstation_id' not in bookings.columns:
        return False, pd.DataFrame()
    bookings = ensure_string_columns(bookings, ['workstation_id','start_time','end_time','date'])
    start_h, start_m = map(int, start_time.split(':'))
    req_start = start_h*60 + start_m
    req_end = req_start + duration_minutes
    day_ws = bookings[(bookings['date'] == meeting_date) & (bookings['workstation_id'].str.upper() == workstation_id.upper())].copy()
    if day_ws.empty:
        return False, pd.DataFrame()
    # Prepare minutes (use .loc to avoid SettingWithCopyWarning)
    if 'start_minutes' not in day_ws.columns:
        day_ws.loc[:, 'start_minutes'] = day_ws['start_time'].str.split(':').str[0].astype(int)*60 + day_ws['start_time'].str.split(':').str[1].astype(int)
    if 'end_minutes' not in day_ws.columns:
        if 'duration_minutes' in day_ws.columns:
            day_ws.loc[:, 'end_minutes'] = day_ws['start_minutes'] + day_ws['duration_minutes']
        elif 'end_time' in day_ws.columns:
            day_ws.loc[:, 'end_minutes'] = day_ws['end_time'].str.split(':').str[0].astype(int)*60 + day_ws['end_time'].str.split(':').str[1].astype(int)
        else:
            day_ws.loc[:, 'end_minutes'] = day_ws['start_minutes']  # fallback
    conflicts = day_ws[day_ws.apply(lambda r: _time_overlap(req_start, req_end, r['start_minutes'], r['end_minutes']), axis=1)]
    return not conflicts.empty, conflicts

def room_has_conflict(bookings: pd.DataFrame, room_id: str, meeting_date: str, start_time: str, duration_minutes: int) -> bool:
    """Check if a room already has an overlapping booking for given window."""
    if bookings.empty or 'room_id' not in bookings.columns:
        return False
    bookings = ensure_string_columns(bookings, ['room_id','start_time','end_time','date'])
    s_h, s_m = map(int, start_time.split(':'))
    req_start = s_h*60 + s_m
    req_end = req_start + duration_minutes
    room_day = bookings[(bookings['date'] == meeting_date) & (bookings['room_id'].astype(str) == str(room_id))].copy()
    if room_day.empty:
        return False
    if 'start_minutes' not in room_day.columns:
        room_day.loc[:, 'start_minutes'] = room_day['start_time'].str.split(':').str[0].astype(int)*60 + room_day['start_time'].str.split(':').str[1].astype(int)
    if 'end_minutes' not in room_day.columns:
        if 'duration_minutes' in room_day.columns:
            room_day.loc[:, 'end_minutes'] = room_day['start_minutes'] + pd.to_numeric(room_day['duration_minutes'], errors='coerce').fillna(0).astype(int)
        elif 'end_time' in room_day.columns:
            room_day.loc[:, 'end_minutes'] = room_day['end_time'].str.split(':').str[0].astype(int)*60 + room_day['end_time'].str.split(':').str[1].astype(int)
        else:
            room_day.loc[:, 'end_minutes'] = room_day['start_minutes']
    return room_day.apply(lambda r: _time_overlap(req_start, req_end, r['start_minutes'], r['end_minutes']), axis=1).any()

def generate_sequential_booking_id(existing: pd.DataFrame) -> str:
    """Generate next sequential booking_id in format BK-000001 based on existing DataFrame."""
    prefix = "BK-"
    if 'booking_id' not in existing.columns or existing.empty:
        return f"{prefix}000001"
    nums = []
    for bid in existing['booking_id'].astype(str):
        if bid.startswith(prefix):
            tail = bid[len(prefix):]
            if tail.isdigit():
                nums.append(int(tail))
    next_num = (max(nums) + 1) if nums else 1
    return f"{prefix}{next_num:06d}"

def append_booking_record(bookings_path: str, room_id: str, workstation_id: str, meeting_date: str, start_time: str, duration_minutes: int, required_capacity: int, source: str = 'app') -> Tuple[bool, str]:
    """Append a booking row with locking. Returns (success, message)."""
    try:
        with _acquire_lock(BOOKING_LOCK_FILE):
            try:
                existing = pd.read_csv(bookings_path)
            except Exception as e:
                return False, f"Failed to read bookings file: {e}"
            existing = ensure_string_columns(existing, ['workstation_id','room_id','date','start_time','end_time','booking_id'])
            for col, default in [
                ('workstation_id', ''),
                ('booking_created_at', ''),
                ('booking_source', '')
            ]:
                if col not in existing.columns:
                    existing[col] = default
            # Sequential booking id generation replaces UUID usage
            booking_id = generate_sequential_booking_id(existing)
            end_time = _compute_end_time(start_time, duration_minutes)
            # Room conflict check (simplified access)
            if 'room_id' in existing.columns:
                room_day = existing[(existing['room_id'].astype(str) == str(room_id)) & (existing['date'] == meeting_date)].copy()
            else:
                room_day = pd.DataFrame()
            def _row_minutes(df: pd.DataFrame) -> pd.DataFrame:
                df = df.copy()
                if df.empty:
                    return df
                if 'start_minutes' not in df.columns:
                    df.loc[:, 'start_minutes'] = df['start_time'].str.split(':').str[0].astype(int)*60 + df['start_time'].str.split(':').str[1].astype(int)
                if 'end_minutes' not in df.columns:
                    if 'duration_minutes' in df.columns:
                        df.loc[:, 'end_minutes'] = df['start_minutes'] + pd.to_numeric(df['duration_minutes'], errors='coerce').fillna(0).astype(int)
                    elif 'end_time' in df.columns:
                        df.loc[:, 'end_minutes'] = df['end_time'].str.split(':').str[0].astype(int)*60 + df['end_time'].str.split(':').str[1].astype(int)
                    else:
                        df.loc[:, 'end_minutes'] = df['start_minutes']
                return df
            room_day = _row_minutes(room_day)
            s_h, s_m = map(int, start_time.split(':'))
            req_start = s_h*60 + s_m
            req_end = req_start + duration_minutes
            if not room_day.empty:
                room_conflict = room_day[room_day.apply(lambda r: _time_overlap(req_start, req_end, r['start_minutes'], r['end_minutes']), axis=1)]
                if not room_conflict.empty:
                    return False, f"Room {room_id} already booked for overlapping time. Conflicts: {room_conflict['booking_id'].tolist()}"
            if 'workstation_id' in existing.columns:
                ws_day = existing[(existing['date'] == meeting_date) & (existing['workstation_id'].str.upper() == workstation_id.upper())].copy()
                ws_day = _row_minutes(ws_day)
                if not ws_day.empty:
                    ws_conflict = ws_day[ws_day.apply(lambda r: _time_overlap(req_start, req_end, r['start_minutes'], r['end_minutes']), axis=1)]
                    if not ws_conflict.empty:
                        return False, f"Workstation {workstation_id} already has a booking in this slot."
            new_row = {
                'booking_id': booking_id,
                'room_id': room_id,
                'date': meeting_date,
                'start_time': start_time,
                'duration_minutes': duration_minutes,
                'end_time': end_time,
                'booked_attendees': required_capacity,
                'actual_attendees': 0,
                'booking_lead_time_hours': 0,
                'cancelled': False,
                'no_show': False,
                'workstation_id': workstation_id,
                'booking_created_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                'booking_source': source
            }
            for k in new_row.keys():
                if k not in existing.columns:
                    existing[k] = pd.NA
            existing = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
            tmp_path = bookings_path + '.tmp'
            existing.to_csv(tmp_path, index=False)
            os.replace(tmp_path, bookings_path)
        return True, f"Booking created (ID: {booking_id})"
    except Exception as e:
        return False, f"Failed to create booking: {e}"

# --- Helper to render booking actions BEFORE floor plan ---
def render_booking_actions(params: Dict):
    """Render booking actions UI (conflict status + booking button) before floor plan."""
    # Live reload for up-to-date conflicts
    bookings_live = load_bookings(BOOKINGS_CSV)
    ws_conflict, ws_conflict_rows = workstation_has_conflict(
        bookings_live,
        params['workstation_id'],
        params['meeting_date'],
        params['start_time'],
        params['duration_minutes']
    )
    room_conflict = room_has_conflict(
        bookings_live,
        params['room_id'],
        params['meeting_date'],
        params['start_time'],
        params['duration_minutes']
    )
    # Style only booking actions button (scoped CSS)
    st.markdown(
        """
        <style>
        .booking-actions .stButton > button {
            background-color:#d32f2f;
            color:#ffffff;
            font-weight:600;
            border:none;
        }
        .booking-actions .stButton > button:hover {
            background-color:#b71c1c;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    with st.container():
        st.subheader("Booking Actions")
        conflict_msgs = []
        if ws_conflict:
            conflict_msgs.append(f"Workstation {params['workstation_id']} already booked for this slot.")
        if room_conflict:
            conflict_msgs.append(f"Room {params['room_id']} is no longer available (recent booking conflict).")
        if conflict_msgs:
            st.warning(" \n".join(conflict_msgs))
            if ws_conflict_rows is not None and not ws_conflict_rows.empty:
                st.info(f"Existing booking IDs: {ws_conflict_rows.get('booking_id', pd.Series([])).tolist()}")
        else:
            disabled_flag = st.session_state.get('booking_success', False)
            # Scoped div for CSS
            with st.container():
                st.markdown('<div class="booking-actions">', unsafe_allow_html=True)
                unique_btn_key = f"book_meeting_room_{params['room_id']}_{params['meeting_date']}_{params['start_time']}"
                if st.button("Book Meeting Room", disabled=disabled_flag, key=unique_btn_key):
                    success, msg = append_booking_record(
                        BOOKINGS_CSV,
                        params['room_id'],
                        params['workstation_id'],
                        params['meeting_date'],
                        params['start_time'],
                        params['duration_minutes'],
                        params['required_capacity']
                    )
                    if success:
                        st.success(msg)
                        st.session_state.booking_success = True
                        # Invalidate cached bookings so subsequent conflict checks are fresh
                        load_bookings.clear()
                    else:
                        st.error(msg)
                st.markdown('</div>', unsafe_allow_html=True)
            if disabled_flag:
                st.info("Booking already completed for this selection.")

if st.button("Find Room"):
    if start_time >= end_time:
        st.error("End time must be after start time!")
    elif meeting_date is None:
        st.error("Please select a valid weekday (Monday-Friday) before proceeding.")
    else:
        duration_minutes, _ = calculate_duration(start_time, end_time)
        rooms = load_rooms(MEETING_ROOMS_CSV)
        workstations = load_workstations(WORKSTATIONS_CSV)
        bookings = load_bookings(BOOKINGS_CSV)
        if rooms.empty or workstations.empty or bookings.empty:
            st.error("Data loading failed; cannot proceed with room selection.")
        else:
            try:
                assert {"start_dt", "end_dt", "date", "start_time"}.issubset(bookings.columns)
                assert bookings["duration_minutes"].gt(0).all()
                assert not bookings["booking_id"].duplicated().any()
            except AssertionError as e:
                st.error(f"Data integrity check failed: {e}")
            else:
                validation_result = validate_inputs(rooms, workstations, workstation_id, required_capacity, meeting_date, start_time, duration_minutes)
                if not validation_result["valid"]:
                    st.error("Input validation failed: " + "; ".join(validation_result["errors"]))
                else:
                    # NEW: Workstation conflict short-circuit BEFORE room suggestion
                    ws_conflict_now, ws_conflict_rows_now = workstation_has_conflict(
                        bookings,
                        workstation_id,
                        meeting_date,
                        start_time,
                        duration_minutes
                    )
                    if ws_conflict_now and not ws_conflict_rows_now.empty:
                        st.warning(f"Workstation {workstation_id} already booked for this slot. Existing booking(s) shown below.")
                        # Show existing booking details table for this workstation/time window
                        display_cols = [c for c in ["booking_id","room_id","date","start_time","end_time","duration_minutes","booked_attendees"] if c in ws_conflict_rows_now.columns]
                        # Ensure end_time present for legacy rows
                        if "end_time" not in ws_conflict_rows_now.columns:
                            ws_conflict_rows_now = ws_conflict_rows_now.copy()
                            ws_conflict_rows_now["end_time"] = ws_conflict_rows_now.apply(lambda r: _compute_end_time(r["start_time"], int(r["duration_minutes"])) if pd.notna(r.get("duration_minutes")) else "", axis=1)
                        st.dataframe(ws_conflict_rows_now[display_cols].sort_values("start_time"))
                        # Highlight first conflicting room on floorplan if available
                        conflict_room_id = ws_conflict_rows_now.iloc[0]["room_id"] if "room_id" in ws_conflict_rows_now.columns else None
                        highlight_room_df = rooms[rooms["id"] == conflict_room_id]
                        if not highlight_room_df.empty:
                            highlight_room = highlight_room_df.iloc[0]
                        else:
                            highlight_room = None
                        plot_floorplan(rooms, workstations, highlight_ws_id=workstation_id, highlight_room=highlight_room)
                        # Clear any previous selection & prevent booking actions
                        st.session_state.pop('selected_room_result', None)
                        st.session_state.pop('selection_params', None)
                        st.session_state.pop('booking_success', None)
                        st.session_state.from_find_room_click = True  # suppress persistent section
                    else:
                        if enable_detailed_logging:
                            st.info(f"Detailed logging enabled (Level: {log_level}) - Check console for reasoning details")
                        result = select_room_strict(
                            rooms, workstations, bookings,
                            required_capacity, workstation_id,
                            meeting_date, start_time, duration_minutes,
                            utilization_threshold=utilization_threshold,
                            distance_tolerance_m=distance_tolerance_m,
                            enable_logging=enable_detailed_logging,
                            log_level=log_level
                        )

                        if result["selected_room"] is None:
                            st.error(result["decision_explanation"])
                            st.session_state.pop('selected_room_result', None)
                            st.session_state.pop('selection_params', None)
                        else:
                            st.success("Suggested Meeting Room:")
                            if result["selected_room"]:
                                room = result["selected_room"]
                                plain_text = (
                                    f"Meeting Room ID: {room['id']} \n"
                                    f"Meeting Room Name: {room['name']} \n"
                                    f"Capacity: {room['capacity']}\n"
                                    f"Distance: {room['distance_m']} meters\n"
                                    f"Utilization: {room['utilization_pct']}%\n"
                                    f"Available: {room['available']}\n\n"
                                    f"Explanation: {result['decision_explanation']}"
                                )
                                st.text(plain_text)
                            # Persist context BEFORE rendering booking actions & floor plan
                            st.session_state.selected_room_result = result
                            st.session_state.selection_params = {
                                'workstation_id': workstation_id,
                                'meeting_date': meeting_date,
                                'start_time': start_time,
                                'duration_minutes': duration_minutes,
                                'required_capacity': required_capacity,
                                'room_id': result['selected_room']['id']
                            }
                            st.session_state.rooms_snapshot = rooms
                            st.session_state.workstations_snapshot = workstations
                            st.session_state.pop('booking_success', None)
                            st.session_state.from_find_room_click = True  # mark this run to suppress persistent duplicate
                            # Booking actions come directly after explanation (before floor plan)
                            render_booking_actions(st.session_state.selection_params)
                            highlight_room = rooms[rooms["id"] == result["selected_room"]["id"]].iloc[0]
                            plot_floorplan(rooms, workstations, highlight_ws_id=workstation_id, highlight_room=highlight_room)

# --- Persistent booking section (reordered: still shows explanation, booking actions before floor plan on reruns) ---
if 'selected_room_result' in st.session_state and st.session_state.get('selected_room_result', {}).get('selected_room') and not st.session_state.get('from_find_room_click', False):
    # Use stored snapshots for plotting
    params = st.session_state.selection_params
    rooms_snap = st.session_state.get('rooms_snapshot')
    workstations_snap = st.session_state.get('workstations_snapshot')
    result = st.session_state.selected_room_result
    if rooms_snap is not None and workstations_snap is not None:
        room = result['selected_room']
        st.text(
            f"Meeting Room ID: {room['id']} \n"
            f"Meeting Room Name: {room['name']} \n"
            f"Capacity: {room['capacity']}\n"
            f"Distance: {room['distance_m']} meters\n"
            f"Utilization: {room['utilization_pct']}%\n"
            f"Available: {room['available']}\n\n"
            f"Explanation: {result['decision_explanation']}"
        )
        render_booking_actions(params)
        highlight_room = rooms_snap[rooms_snap["id"] == room["id"]].iloc[0]
        plot_floorplan(rooms_snap, workstations_snap, highlight_ws_id=params['workstation_id'], highlight_room=highlight_room)