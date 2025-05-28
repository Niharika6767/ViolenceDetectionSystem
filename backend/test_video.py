from app import video_to_frames  # Ensure app.py is in the same directory

frames = video_to_frames(r"C:\Users\anshu\Downloads\4761738-uhd_4096_2160_25fps.mp4")
print("Number of frames extracted:", len(frames))
