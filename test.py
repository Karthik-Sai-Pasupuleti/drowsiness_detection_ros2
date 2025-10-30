import cv2

# Path to your video file
video_path = 'drowsiness_data/pilot_test_indrajith/videos/full_video.mp4'

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through frames
while True:
    ret, frame = cap.read()

    # If frame read was not successful, break the loop
    if not ret:
        print("Reached end of video or failed to read frame.")
        break

    # Display the frame
    cv2.imshow('Video', frame)

    # Press 'q' to quit early
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
