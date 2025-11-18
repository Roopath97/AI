import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from typing import Tuple, Dict
from datetime import datetime

# -------- File Paths --------
import os
_d = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MEETING_ROOMS_CSV, WORKSTATIONS_CSV, BOOKINGS_CSV = [os.path.join(_d, f) for f in ["meeting_rooms.csv", "workstations.csv", "bookings.csv"]]

# ---------------- Utility Functions ---------------- #

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df

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
        return exact, "Exact capacity tier applied"
    over = rooms[rooms["capacity"] > required_capacity]
    if over.empty:
        return pd.DataFrame(), "No rooms meet capacity requirement"
    min_over = over["capacity"].min()
    return over[over["capacity"] == min_over], f"Minimal over-capacity tier applied (capacity={min_over})"

def filter_available(rooms: pd.DataFrame, bookings: pd.DataFrame, meeting_date: str, start_min: int, end_min: int) -> pd.DataFrame:
    available_rooms = []
    for _, room in rooms.iterrows():
        room_bookings = bookings[(bookings["room_id"] == room["id"]) & (bookings["date"] == meeting_date)]
        conflict = False
        for _, b in room_bookings.iterrows():
            # Check if booking is active (not cancelled and confirmed)
            if not b["cancelled"]:  # Adjust if you have a 'confirmed' column
                bh, bm = map(int, b["start_time"].split(":"))
                b_start = bh * 60 + bm
                b_end = b_start + b["duration_minutes"]
                if is_overlap(start_min, end_min, b_start, b_end):
                    conflict = True
                    break
        if not conflict:
            available_rooms.append(room)
    return pd.DataFrame(available_rooms)

def compute_distances(rooms: pd.DataFrame, ws: pd.DataFrame, workstation_id: str) -> pd.DataFrame:
    ws_row = ws[ws["workstation_id"].str.upper() == workstation_id.upper()]
    if ws_row.empty:
        raise ValueError(f"Workstation {workstation_id} not found")
    ws_x, ws_y = ws_row.iloc[0]["x"], ws_row.iloc[0]["y"]
    rooms = rooms.copy()
    rooms["distance_m"] = ((rooms["center_x_calc"] - ws_x)**2 + (rooms["center_y_calc"] - ws_y)**2).apply(math.sqrt)
    return rooms

def compute_daily_utilization(rooms: pd.DataFrame, bookings: pd.DataFrame, meeting_date: str) -> pd.DataFrame:
    rooms = rooms.copy()
    utilization_map = {}
    for _, room in rooms.iterrows():
        room_bookings = bookings[(bookings["room_id"] == room["id"]) & (bookings["date"] == meeting_date) & (~bookings["cancelled"])]
        total_minutes = room_bookings["duration_minutes"].sum()
        utilization_map[room["id"]] = total_minutes / 480.0
    rooms["utilization"] = rooms["id"].map(utilization_map).fillna(0.0)
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
    distance_tolerance_m: float = 5.0
) -> Dict:
    rooms = compute_room_centers(normalize_columns(rooms))
    workstations = normalize_columns(workstations)
    bookings = normalize_columns(bookings)

    meeting_date, start_min, end_min = get_meeting_window(meeting_date, start_time, duration_minutes)

    capacity_filtered, capacity_explanation = filter_capacity_tier(rooms, required_capacity)
    if capacity_filtered.empty:
        return {"selected_room": None, "decision_explanation": capacity_explanation}

    available_rooms = filter_available(capacity_filtered, bookings, meeting_date, start_min, end_min)
    if available_rooms.empty:
        return {"selected_room": None, "decision_explanation": "All eligible rooms unavailable"}

    available_rooms = compute_distances(available_rooms, workstations, workstation_id)
    available_rooms = compute_daily_utilization(available_rooms, bookings, meeting_date)

    available_rooms = available_rooms.sort_values(by=["distance_m", "utilization", "id"])
    nearest_room = available_rooms.iloc[0]
    d_min = nearest_room["distance_m"]

    explanation = [capacity_explanation]
    if nearest_room["utilization"] > utilization_threshold:
        explanation.append(f"Nearest room utilization {nearest_room['utilization']:.2f} > threshold")
        candidates = available_rooms[(available_rooms["distance_m"] <= d_min + distance_tolerance_m) &
                                     (available_rooms["utilization"] < nearest_room["utilization"])]
        if not candidates.empty:
            candidates = candidates.sort_values(by=["utilization", "distance_m", "id"])
            selected = candidates.iloc[0]
            explanation.append("Tolerance triggered: selected lower-utilization alternative")
        else:
            selected = nearest_room
            explanation.append("No better alternative found within tolerance")
    else:
        selected = nearest_room
        explanation.append("Nearest room under utilization threshold")


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

time_options = generate_time_options()
default_start_index = get_default_time_index()

workstation_id = st.text_input("Workstation ID", "WS-11")
required_capacity = st.number_input("Required Capacity", min_value=1, value=4)
meeting_date = st.date_input("Meeting Date").strftime("%Y-%m-%d")

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
            st.info(f"📅 **Total booking duration:** {int(duration_hours)} hour(s){next_day_text}")
        else:
            st.info(f"📅 **Total booking duration:** {duration_hours:.1f} hours ({duration_minutes} minutes){next_day_text}")
    else:
        st.warning("⚠️ End time must be after start time!")
else:
    st.warning("⚠️ Please select valid start and end times!")

if st.button("Find Room"):
    if start_time >= end_time:
        st.error("End time must be after start time!")
    else:
        duration_minutes, _ = calculate_duration(start_time, end_time)
        
        rooms = pd.read_csv(MEETING_ROOMS_CSV)
        workstations = pd.read_csv(WORKSTATIONS_CSV)
        bookings = pd.read_csv(BOOKINGS_CSV)

        rooms = compute_room_centers(normalize_columns(rooms))
        workstations = normalize_columns(workstations)

        result = select_room_strict(
            rooms, workstations, bookings,
            required_capacity, workstation_id,
            meeting_date, start_time, duration_minutes
        )

        if result["selected_room"] is None:
            st.error(result["decision_explanation"])
        else:
            st.success("✅ Suggested Meeting Room:")
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