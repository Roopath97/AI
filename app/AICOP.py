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

    # Normalize booleans (if strings TRUE/FALSE)
    bool_cols = ["is_recurring", "has_catering", "has_av_equipment", "cancelled", "no_show"]
    for c in bool_cols:
        if c in b.columns:
            b[c] = b[c].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
            b[c] = b[c].fillna(False)

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
    enable_logging: bool = False
) -> Dict:
    # Configure logger for this session
    global room_logger
    room_logger = setup_room_selection_logger(enable_logging)
    
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

if st.button("Find Room"):
    if start_time >= end_time:
        st.error("End time must be after start time!")
    elif meeting_date is None:
        st.error("Please select a valid weekday (Monday-Friday) before proceeding.")
    else:
        duration_minutes, _ = calculate_duration(start_time, end_time)
        
        rooms = pd.read_csv(MEETING_ROOMS_CSV)
        workstations = pd.read_csv(WORKSTATIONS_CSV)
        bookings = pd.read_csv(BOOKINGS_CSV)

        rooms = compute_room_centers(normalize_columns(rooms))
        workstations = normalize_columns(workstations)        # PREPROCESS BOOKINGS HERE (dedupe + invalid window cleanup)
        bookings = preprocess_bookings_raw(normalize_columns(bookings))

        assert {"start_dt", "end_dt", "date", "start_time"}.issubset(bookings.columns)
        assert bookings["duration_minutes"].gt(0).all()
        assert not bookings["booking_id"].duplicated().any()

        validation_result = validate_inputs(rooms, workstations, workstation_id, required_capacity, meeting_date, start_time, duration_minutes)
        if not validation_result["valid"]:
            st.error("Input validation failed: " + "; ".join(validation_result["errors"]))
        else:
            # Show logging status if enabled
            if enable_detailed_logging:
                st.info(f"Detailed logging enabled (Level: {log_level}) - Check console for reasoning details")
            
            result = select_room_strict(
                rooms, workstations, bookings,
                required_capacity, workstation_id,
                meeting_date, start_time, duration_minutes,
                enable_logging=enable_detailed_logging
            )

            if result["selected_room"] is None:
                st.error(result["decision_explanation"])
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

                # Highlight selected room and workstation on floor plan
                highlight_room = rooms[rooms["id"] == result["selected_room"]["id"]].iloc[0]
                plot_floorplan(rooms, workstations, highlight_ws_id=workstation_id, highlight_room=highlight_room)