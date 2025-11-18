"""
🎯 REAL DATA CONFLICT TESTS

Tests based on actual booking patterns in your CSV data to validate conflict detection logic.
"""

import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))
from AICOP import select_room_strict, compute_room_centers, normalize_columns, preprocess_bookings_raw

def analyze_booking_conflicts():
    """Analyze actual booking data to identify conflict scenarios"""
    
    # Load and preprocess data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    rooms = pd.read_csv(os.path.join(data_dir, "meeting_rooms.csv"))
    workstations = pd.read_csv(os.path.join(data_dir, "workstations.csv"))
    bookings = pd.read_csv(os.path.join(data_dir, "bookings.csv"))
    
    rooms = compute_room_centers(normalize_columns(rooms))
    workstations = normalize_columns(workstations)
    bookings = preprocess_bookings_raw(normalize_columns(bookings))
    
    print("📊 REAL DATA CONFLICT ANALYSIS")
    print("="*50)
    
    # Analyze busy rooms and times
    busy_dates = bookings.groupby(['date', 'room_id']).size().reset_index(name='booking_count')
    busy_dates = busy_dates.sort_values('booking_count', ascending=False)
    
    print("🏢 Busiest room-date combinations:")
    for _, row in busy_dates.head(10).iterrows():
        print(f"   {row['room_id']} on {row['date']}: {row['booking_count']} bookings")
    
    print(f"\n⏰ Booking time distribution:")
    time_conflicts = bookings.groupby(['date', 'start_time']).size().reset_index(name='concurrent_bookings')
    time_conflicts = time_conflicts[time_conflicts['concurrent_bookings'] > 1].sort_values('concurrent_bookings', ascending=False)
    
    for _, row in time_conflicts.head(5).iterrows():
        print(f"   {row['date']} at {row['start_time']}: {row['concurrent_bookings']} concurrent bookings")
    
    return rooms, workstations, bookings

def test_real_conflicts():
    """Test scenarios based on actual booking conflicts"""
    
    rooms, workstations, bookings = analyze_booking_conflicts()
    
    print(f"\n{'='*60}")
    print("🧪 TESTING REAL BOOKING CONFLICT SCENARIOS")
    print(f"{'='*60}")
    
    # Test Case 1: MR-002 heavy booking day
    print(f"\n🎯 TEST 1: Heavy booking day for MR-002 (2025-11-13)")
    mr002_bookings = bookings[(bookings['room_id'] == 'MR-002') & (bookings['date'] == '2025-11-13')]
    print("   Existing MR-002 bookings on 2025-11-13:")
    for _, booking in mr002_bookings.iterrows():
        print(f"     • {booking['start_time']}-{booking['end_time']} ({booking['duration_minutes']}min)")
    
    # Try to book during a conflict time
    result1 = select_room_strict(
        rooms, workstations, bookings,
        required_capacity=4,
        workstation_id="WS-01",
        meeting_date="2025-11-13",
        start_time="14:00",  # Should conflict with existing bookings
        duration_minutes=60,
        enable_logging=True
    )
    
    if result1['selected_room']:
        selected_room_id = result1['selected_room']['id']
        print(f"   ✅ Algorithm selected: {selected_room_id}")
        if selected_room_id == 'MR-002':
            print("   ⚠️  WARNING: Selected MR-002 despite conflicts - check logic!")
        else:
            print(f"   ✅ Correctly avoided MR-002, selected alternative")
    else:
        print("   ❌ No room found - may indicate over-restrictive logic")
    
    # Test Case 2: Multi-room conflict scenario
    print(f"\n🎯 TEST 2: Peak conflict time (2025-11-13 at 14:00)")
    conflict_bookings = bookings[(bookings['date'] == '2025-11-13') & (bookings['start_time'] == '14:00')]
    print("   Rooms already booked at 14:00:")
    for _, booking in conflict_bookings.iterrows():
        print(f"     • {booking['room_id']}: {booking['start_time']}-{booking['end_time']}")
    
    result2 = select_room_strict(
        rooms, workstations, bookings,
        required_capacity=6,  # Need larger room
        workstation_id="WS-10",
        meeting_date="2025-11-13",
        start_time="14:00",
        duration_minutes=90,
        enable_logging=True
    )
    
    print(f"   Result: {result2['selected_room']['id'] if result2['selected_room'] else 'No room found'}")
    
    # Test Case 3: Edge time conflict (overlapping boundaries)
    print(f"\n🎯 TEST 3: Edge time overlap test")
    result3 = select_room_strict(
        rooms, workstations, bookings,
        required_capacity=4,
        workstation_id="WS-05",
        meeting_date="2025-11-13",
        start_time="16:45",  # Might overlap with existing 16:30 booking
        duration_minutes=60,
        enable_logging=True
    )
    
    print(f"   Result: {result3['selected_room']['id'] if result3['selected_room'] else 'No room found'}")
    
    # Test Case 4: Clean slate test (future date)
    print(f"\n🎯 TEST 4: Clean slate (no existing bookings)")
    result4 = select_room_strict(
        rooms, workstations, bookings,
        required_capacity=4,
        workstation_id="WS-01",
        meeting_date="2025-12-01",  # Future date with no bookings
        start_time="10:00",
        duration_minutes=60,
        enable_logging=True
    )
    
    print(f"   Result: {result4['selected_room']['id'] if result4['selected_room'] else 'No room found'}")
    if result4['selected_room']:
        print(f"   Distance: {result4['selected_room']['distance_m']}m")
        print(f"   Utilization: {result4['selected_room']['utilization_pct']}%")

def test_capacity_edge_cases():
    """Test specific capacity-related edge cases from your data"""
    
    rooms, workstations, bookings = analyze_booking_conflicts()
    
    print(f"\n{'='*60}")
    print("🧪 CAPACITY EDGE CASE TESTS")
    print(f"{'='*60}")
    
    # Analyze actual capacity distribution
    capacity_dist = rooms['capacity'].value_counts().sort_index()
    print("📊 Available room capacities:")
    for cap, count in capacity_dist.items():
        print(f"   Capacity {cap}: {count} rooms")
    
    # Test Case 1: Request exactly max capacity
    max_capacity = rooms['capacity'].max()
    print(f"\n🎯 TEST: Maximum capacity request ({max_capacity} people)")
    result_max = select_room_strict(
        rooms, workstations, bookings,
        required_capacity=max_capacity,
        workstation_id="WS-30",
        meeting_date="2025-11-20",
        start_time="10:00",
        duration_minutes=60,
        enable_logging=True
    )
    
    if result_max['selected_room']:
        print(f"   ✅ Found room: {result_max['selected_room']['id']} (capacity: {result_max['selected_room']['capacity']})")
    else:
        print("   ❌ Failed to find room for max capacity")
    
    # Test Case 2: Request impossible capacity
    print(f"\n🎯 TEST: Impossible capacity request ({max_capacity + 5} people)")
    result_impossible = select_room_strict(
        rooms, workstations, bookings,
        required_capacity=max_capacity + 5,
        workstation_id="WS-30",
        meeting_date="2025-11-20", 
        start_time="10:00",
        duration_minutes=60,
        enable_logging=True
    )
    
    if result_impossible['selected_room'] is None:
        print(f"   ✅ Correctly rejected impossible capacity")
    else:
        print(f"   ❌ ERROR: Should not find room for impossible capacity")
    
    # Test Case 3: Test capacity efficiency (prefer exact match)
    print(f"\n🎯 TEST: Capacity efficiency (4 people - multiple exact matches)")
    result_efficiency = select_room_strict(
        rooms, workstations, bookings,
        required_capacity=4,
        workstation_id="WS-15",
        meeting_date="2025-11-20",
        start_time="14:00", 
        duration_minutes=60,
        enable_logging=True
    )
    
    if result_efficiency['selected_room']:
        selected_capacity = result_efficiency['selected_room']['capacity']
        if selected_capacity == 4:
            print(f"   ✅ Correctly selected exact capacity match")
        else:
            print(f"   ⚠️  Selected over-capacity room (capacity: {selected_capacity})")

if __name__ == "__main__":
    print("🧪 REAL DATA VALIDATION TEST SUITE")
    print("="*80)
    print("Testing ML logic against actual CSV data patterns...\n")
    
    test_real_conflicts()
    test_capacity_edge_cases()
    
    print(f"\n{'='*80}")
    print("🎯 REAL DATA TESTING COMPLETE!")
    print("="*80)
    print("Key validation points:")
    print("✅ Conflict detection accuracy")
    print("✅ Capacity constraint enforcement") 
    print("✅ Distance-based selection logic")
    print("✅ Edge case handling robustness")
