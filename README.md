**Smart Meeting Room Selector**
A Streamlit app that recommends the best available meeting room near a given workstation, based on capacity, availability, walking distance, and daily utilization. It also renders a simple floor plan to visualize rooms and workstations, highlighting the suggested room and selected workstation.

**Why this app**
Reduce time to book: Quickly find an appropriate room without scanning calendars.
Improve fairness: Avoid overloading popular rooms with a utilization-aware policy.
Visual context: See the floor layout, room capacities, and your selected workstation.
Features
Capacity tiering: Prefers an exact capacity match; otherwise selects the minimal over-capacity tier.
Time-window availability: Excludes rooms with overlapping bookings in the requested window.
Distance-based ranking: Chooses the nearest available room to the specified workstation.
Utilization-aware selection: If the nearest room is highly utilized, searches within a small distance tolerance for a lower-utilization alternative.
Floor plan visualization: Plots rooms color-coded by capacity and highlights the recommended room and selected workstation.
Transparent decisions: Returns a clear explanation of why a room was chosen.
App demo (what you’ll see)
Inputs for workstation ID, capacity, date, time, and duration
A “Find Room” button to compute the suggestion
A textual summary of the recommended room (ID, name, capacity, distance, utilization, availability)
A floor plan figure with:
Colored room rectangles (by capacity)
The selected workstation
A highlight ring around the recommended room
Project structure
app.py (your Streamlit script with selection logic and plotting)
meeting_rooms.csv
workstations1.csv
bookings.csv
README.md (this file)
You can rename the script file if preferred; update run instructions accordingly.

**Installation**
Python 3.9+ recommended
Install dependencies:

pip install streamlit pandas matplotlib

**Running the app**
Place the CSV files in the same directory as the script.
From a terminal in that directory:
streamlit run app.py
Streamlit will open a browser tab. If it doesn’t, follow the terminal URL.

**Configuration and data**
The script expects these files and column schemas. Column names are normalized to lowercase and trimmed.

meeting_rooms.csv: id, name, x, y, width, height, capacity
workstations1.csv: workstation_id, x, y
bookings.csv: room_id, date (YYYY-MM-DD), start_time (HH:MM, 24-hour), duration_minutes (int), cancelled (boolean)
Coordinates are treated as meters for display and distance.

**How the selection works**
Capacity tiering: filter rooms to exact capacity; if none, choose the smallest capacity greater than required.
Availability: remove rooms with overlapping, non-cancelled bookings on the selected date.
Distance: compute Euclidean distance from the workstation to each candidate room’s center.
Utilization: compute day utilization per room as booked_minutes/480 (8-hour day).
Tolerance: if the nearest room’s utilization exceeds a threshold (default 0.65), search within distance_tolerance_m (default 5.0 meters) for rooms with lower utilization and pick the best among them; otherwise keep the nearest.
Returned result example:

selected_room: id, name, capacity, distance_m, utilization_pct, available
decision_explanation: semicolon-joined rationale string
Code reference
Core functions contained in app.py:

normalize_columns(df): lowercase and trim columns
compute_room_centers(rooms): add center_x_calc, center_y_calc, right, top
is_overlap(a_start, a_end, b_start, b_end): check interval overlap
get_meeting_window(date, start_time, duration_minutes): compute start/end minutes
filter_capacity_tier(rooms, required_capacity): apply capacity rule
filter_available(rooms, bookings, meeting_date, start_min, end_min): filter by window and cancellation
compute_distances(rooms, ws, workstation_id): distance to workstation
compute_daily_utilization(rooms, bookings, meeting_date): utilization fraction
select_room_strict(...): orchestrate the selection and explanation
plot_floorplan(...): matplotlib plot rendered via Streamlit
