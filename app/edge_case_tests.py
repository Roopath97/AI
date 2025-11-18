"""
🧪 EDGE CASE TEST SUITE for Smart Meeting Room Selector

This test suite validates the ML logic with challenging scenarios based on data analysis.
"""

import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add the app directory to Python path
sys.path.append(os.path.dirname(__file__))

from AICOP import (
    select_room_strict,
    compute_room_centers,
    normalize_columns,
    preprocess_bookings_raw
)

class EdgeCaseTestSuite:
    def __init__(self):
        # Load data
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.rooms = pd.read_csv(os.path.join(data_dir, "meeting_rooms.csv"))
        self.workstations = pd.read_csv(os.path.join(data_dir, "workstations.csv"))
        self.bookings = pd.read_csv(os.path.join(data_dir, "bookings.csv"))
        
        # Preprocess
        self.rooms = compute_room_centers(normalize_columns(self.rooms))
        self.workstations = normalize_columns(self.workstations)
        self.bookings = preprocess_bookings_raw(normalize_columns(self.bookings))
        
        print("📊 DATA ANALYSIS SUMMARY:")
        print(f"   • Rooms: {len(self.rooms)} (capacities: {sorted(self.rooms['capacity'].unique())})")
        print(f"   • Workstations: {len(self.workstations)}")
        print(f"   • Bookings: {len(self.bookings)} (dates: {self.bookings['date'].min()} to {self.bookings['date'].max()})")

    def run_test_case(self, test_name, **kwargs):
        """Run a single test case with detailed logging"""
        print(f"\n{'='*60}")
        print(f"🧪 TEST: {test_name}")
        print(f"{'='*60}")
        
        result = select_room_strict(
            self.rooms, self.workstations, self.bookings,
            enable_logging=True,  # Enable detailed logging for all tests
            **kwargs
        )
        
        print(f"\n📋 RESULT: {result['selected_room']['name'] if result['selected_room'] else 'NO ROOM FOUND'}")
        if result['selected_room']:
            room = result['selected_room']
            print(f"   • Room: {room['id']} ({room['name']})")
            print(f"   • Distance: {room['distance_m']}m")
            print(f"   • Utilization: {room['utilization_pct']}%")
            print(f"   • Capacity: {room['capacity']}")
        print(f"   • Explanation: {result['decision_explanation']}")
        return result

    def test_edge_cases(self):
        print("🚀 STARTING EDGE CASE TEST SUITE")
        print("="*80)
        
        # Test Case 1: IMPOSSIBLE CAPACITY REQUIREMENT
        self.run_test_case(
            "IMPOSSIBLE CAPACITY - Requesting 15 people (max room capacity is 8)",
            required_capacity=15,
            workstation_id="WS-01",
            meeting_date="2025-11-19",
            start_time="10:00",
            duration_minutes=60
        )
        
        # Test Case 2: ALL ROOMS BOOKED (create scenario with heavy bookings)
        self.run_test_case(
            "PEAK TIME - All suitable rooms likely booked",
            required_capacity=4,
            workstation_id="WS-01", 
            meeting_date="2025-11-13",  # Date with existing bookings
            start_time="14:00",  # Peak time with conflicts
            duration_minutes=120
        )
        
        # Test Case 3: SINGLE PERSON MEETING (minimal capacity)
        self.run_test_case(
            "MINIMAL CAPACITY - Single person meeting",
            required_capacity=1,
            workstation_id="WS-30",  # Middle workstation
            meeting_date="2025-11-20",
            start_time="09:00",
            duration_minutes=30
        )
        
        # Test Case 4: MAXIMUM CAPACITY MEETING
        self.run_test_case(
            "MAXIMUM CAPACITY - Large meeting (8 people)",
            required_capacity=8,
            workstation_id="WS-01",
            meeting_date="2025-11-20",
            start_time="11:00", 
            duration_minutes=90
        )
        
        # Test Case 5: EDGE WORKSTATION (corner position)
        self.run_test_case(
            "EDGE WORKSTATION - WS-60 (far corner)",
            required_capacity=4,
            workstation_id="WS-60",  # Furthest workstation
            meeting_date="2025-11-20",
            start_time="10:00",
            duration_minutes=60
        )
        
        # Test Case 6: HIGH UTILIZATION THRESHOLD TEST
        self.run_test_case(
            "HIGH UTILIZATION SCENARIO - Force utilization fallback",
            required_capacity=4,
            workstation_id="WS-01",
            meeting_date="2025-11-13",  # Busy day
            start_time="10:00",
            duration_minutes=60,
            utilization_threshold=0.1  # Very low threshold to force fallback
        )
        
        # Test Case 7: VERY SHORT MEETING
        self.run_test_case(
            "MICRO MEETING - 15 minute meeting",
            required_capacity=3,
            workstation_id="WS-15",
            meeting_date="2025-11-20",
            start_time="16:45",
            duration_minutes=15
        )
        
        # Test Case 8: VERY LONG MEETING
        self.run_test_case(
            "MARATHON MEETING - 8 hour meeting",
            required_capacity=6,
            workstation_id="WS-25",
            meeting_date="2025-11-21",
            start_time="09:00",
            duration_minutes=480
        )
        
        # Test Case 9: LATE NIGHT MEETING
        self.run_test_case(
            "LATE NIGHT MEETING - After hours",
            required_capacity=4,
            workstation_id="WS-10",
            meeting_date="2025-11-20",
            start_time="22:00",
            duration_minutes=90
        )
        
        # Test Case 10: WEEKEND MEETING
        # Calculate next weekend
        today = datetime.now()
        days_until_saturday = (5 - today.weekday()) % 7
        if days_until_saturday == 0:
            days_until_saturday = 7
        next_saturday = today + timedelta(days=days_until_saturday)
        
        self.run_test_case(
            "WEEKEND MEETING - Saturday meeting",
            required_capacity=5,
            workstation_id="WS-20",
            meeting_date=next_saturday.strftime("%Y-%m-%d"),
            start_time="10:00",
            duration_minutes=120
        )
        
        # Test Case 11: EXACT CAPACITY MATCH vs OVER-CAPACITY
        self.run_test_case(
            "EXACT CAPACITY CHALLENGE - 4 person meeting (multiple 4-capacity rooms)",
            required_capacity=4,
            workstation_id="WS-05",
            meeting_date="2025-11-20",
            start_time="14:00",
            duration_minutes=90
        )
        
        # Test Case 12: DISTANCE TOLERANCE TEST
        self.run_test_case(
            "DISTANCE TOLERANCE - Tiny tolerance to force nearest room",
            required_capacity=4,
            workstation_id="WS-01",
            meeting_date="2025-11-20",
            start_time="15:00",
            duration_minutes=60,
            distance_tolerance_m=0.1  # Very small tolerance
        )

    def test_data_edge_cases(self):
        """Test edge cases related to data quality and boundaries"""
        print(f"\n{'='*60}")
        print("📊 DATA BOUNDARY TESTS")
        print(f"{'='*60}")
        
        # Test invalid workstation
        try:
            result = select_room_strict(
                self.rooms, self.workstations, self.bookings,
                required_capacity=4,
                workstation_id="WS-999",  # Non-existent
                meeting_date="2025-11-20",
                start_time="10:00",
                duration_minutes=60,
                enable_logging=True
            )
            print("❌ ERROR: Should have failed for invalid workstation")
        except ValueError as e:
            print(f"✅ PASS: Correctly caught invalid workstation: {e}")
        
        # Test boundary room capacities
        print(f"\n🎯 Testing boundary capacities...")
        min_capacity = self.rooms['capacity'].min()  # Should be 3
        max_capacity = self.rooms['capacity'].max()  # Should be 8
        
        print(f"   • Min room capacity: {min_capacity}")
        print(f"   • Max room capacity: {max_capacity}")
        
        # Test just above max capacity
        result = select_room_strict(
            self.rooms, self.workstations, self.bookings,
            required_capacity=max_capacity + 1,
            workstation_id="WS-01",
            meeting_date="2025-11-20",
            start_time="10:00",
            duration_minutes=60,
            enable_logging=True
        )
        
        if result['selected_room'] is None:
            print(f"✅ PASS: Correctly rejected capacity {max_capacity + 1}")
        else:
            print(f"❌ ERROR: Should not find room for capacity {max_capacity + 1}")

def main():
    """Run the complete edge case test suite"""
    test_suite = EdgeCaseTestSuite()
    
    print("🧪 SMART MEETING ROOM SELECTOR - EDGE CASE VALIDATION")
    print("="*80)
    print("This suite tests challenging scenarios to validate ML logic robustness.")
    print()
    
    # Run main edge cases
    test_suite.test_edge_cases()
    
    # Run data boundary tests
    test_suite.test_data_edge_cases()
    
    print(f"\n{'='*80}")
    print("🎯 EDGE CASE TESTING COMPLETE!")
    print("="*80)
    print("Review the logs above to ensure your ML logic handles all scenarios correctly.")
    print("Key things to verify:")
    print("✅ Capacity constraints are respected")
    print("✅ Availability conflicts are detected")
    print("✅ Distance calculations are accurate") 
    print("✅ Utilization thresholds work as expected")
    print("✅ Edge cases fail gracefully with clear messages")

if __name__ == "__main__":
    main()
