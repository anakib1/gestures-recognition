import unittest
import cv2
import mediapipe as mp
import os
from main import check_finger_configuration_C # Assuming main.py is in the same dir

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

class TestHandGestureCFromImages(unittest.TestCase):

    hands_model = None

    @classmethod
    def setUpClass(cls):
        """Initialize the MediaPipe Hands model once for the test class."""
        cls.hands_model = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5)
        print("\nMediaPipe Hands model initialized for static images.")

    @classmethod
    def tearDownClass(cls):
        """Release the MediaPipe Hands model."""
        if cls.hands_model:
            cls.hands_model.close()
            print("\nMediaPipe Hands model closed.")

    def process_image_and_check(self, image_filename, expected_result):
        """Helper function to load, process image, and check gesture."""
        print(f"\nProcessing {image_filename} (Expected: {'Pass' if expected_result else 'Fail'})")
        image_path = os.path.join(os.path.dirname(__file__), image_filename)
        
        if not os.path.exists(image_path):
            self.fail(f"Image file not found: {image_path}")

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            self.fail(f"Failed to read image: {image_path}")

        # Convert the BGR image to RGB and process it with MediaPipe Hands
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands_model.process(image_rgb)

        # Check if landmarks were detected
        if not results.multi_hand_landmarks:
            self.fail(f"No hand landmarks detected in {image_filename}")
            
        # Assuming only one hand is relevant per image
        hand_landmarks_list = results.multi_hand_landmarks
        
        # Check the finger configuration
        actual_result = check_finger_configuration_C(hand_landmarks_list, 0) # Pass the list
        
        # Assert the result
        if expected_result:
            self.assertTrue(actual_result, f"{image_filename} failed: Expected Pass, Got Fail")
        else:
            self.assertFalse(actual_result, f"{image_filename} failed: Expected Fail, Got Pass")

    # --- Individual Tests ---

    def test_image_1_pass(self):
        self.process_image_and_check("samples/1.png", expected_result=True)

    def test_image_2_fail(self):
        self.process_image_and_check("samples/2.png", expected_result=True)

    def test_image_3_pass(self):
        self.process_image_and_check("samples/3.png", expected_result=True)

    def test_image_4_pass(self):
        self.process_image_and_check("samples/4.png", expected_result=True)

    def test_image_5_pass(self):
        self.process_image_and_check("samples/5.png", expected_result=True)

if __name__ == '__main__':
    # Ensure images 1.png, 2.png, 3.png, 4.png, 5.png are in the same directory
    # as this test script (test_main.py)
    unittest.main() 