import time
import math
import taichi as ti
from interstellar_renderer import InterstellarRenderer

def main():
    """
    Initializes the renderer and runs a scripted camera animation loop,
    with options to either display in a GUI or render to a video file.
    """
    # --- 1. RENDER CONFIGURATION ---
    # -- Choose your mode --
    # Mode 1: Real-time preview
    # SHOW_GUI = True
    # SAVE_VIDEO = False
    
    # Mode 2: Offline rendering to video (no GUI)
    SHOW_GUI = False
    SAVE_VIDEO = True

    # -- Video settings (only used if SAVE_VIDEO is True) --
    VIDEO_DURATION_SECONDS = 1  # How long the final video should be. NOTE: change this and the fps to 60 for the full movie
    VIDEO_FPS = 1               # Frames per second
    OUTPUT_FILENAME = "interstellar_flight.mp4"

    # 2. Initialize the scene renderer with the chosen configuration.
    scene = InterstellarRenderer(
        show_gui=SHOW_GUI,
        save_video_path=OUTPUT_FILENAME if SAVE_VIDEO else None,
        video_fps=VIDEO_FPS,
        width=1920
    )
    
    # 3. Main application loop
    if SAVE_VIDEO:
        # --- OFFLINE RENDER LOOP ---
        total_frames = int(VIDEO_DURATION_SECONDS * VIDEO_FPS)
        print(f"Starting offline render of {total_frames} frames to {OUTPUT_FILENAME}...")
        
        for frame in range(total_frames):
            start_frame_time = time.time()
            
            # Calculate animation time based on frame number for deterministic output
            animation_time = frame / VIDEO_FPS
            
            # Get camera vectors for the current time
            cam_pos, cam_fwd, world_up, fov = get_camera_vectors_at_time(animation_time)
            
            # Render the frame
            scene.step(cam_pos, cam_fwd, world_up, fov)
            
            # Progress report
            end_frame_time = time.time()
            frame_duration = end_frame_time - start_frame_time
            print(f"  - Rendered frame {frame + 1}/{total_frames} in {frame_duration:.2f}s")

    else:
        # --- REAL-TIME GUI LOOP ---
        start_time = time.time()
        while scene.gui and scene.gui.running:
            # Handle GUI events
            for e in scene.gui.get_events(ti.GUI.PRESS):
                if e.key == ti.GUI.ESCAPE:
                    scene.gui.running = False

            # Calculate animation time based on wall-clock time
            animation_time = time.time() - start_time

            # Get camera vectors
            cam_pos, cam_fwd, world_up, fov = get_camera_vectors_at_time(animation_time)
            
            # Render and display the frame
            scene.step(cam_pos, cam_fwd, world_up, fov)

    # 4. Clean up and finalize resources
    scene.close()
    if SAVE_VIDEO:
        print(f"Video saved successfully to {OUTPUT_FILENAME}")

def get_camera_vectors_at_time(animation_time):
    """
    Calculates the camera position, direction, up vector, and FOV for a given time.
    This function defines the cinematic camera path.
    """
    # --- Orbiting motion (horizontal angle) ---
    orbit_period = 120.0   # slow orbit to last 2 minutes for smooth effect
    phi = (animation_time / orbit_period) * 2.0 * math.pi + math.pi

    # --- Dolly motion (distance from center) ---
    start_radius, end_radius = 50.0, 18.0
    animation_duration = 60.0   # match the video duration
    progress = min(animation_time / animation_duration, 1.0)
    eased_progress = 0.5 * (1.0 - math.cos(progress * math.pi))
    radius = start_radius + (end_radius - start_radius) * eased_progress
    
    # --- Vertical motion (vertical angle) ---
    start_theta, end_theta = math.pi / 2.0 - 0.4, math.pi / 2.0 + 0.2
    theta = start_theta + (end_theta - start_theta) * eased_progress

    # --- FOV (zoom effect) ---
    start_fov, end_fov = 1.0, 1.3
    fov = start_fov + (end_fov - start_fov) * eased_progress

    # Calculate final camera vectors from path parameters
    cam_x = radius * math.sin(theta) * math.cos(phi)
    cam_y = radius * math.cos(theta)
    cam_z = radius * math.sin(theta) * math.sin(phi)
    
    cam_pos = ti.Vector([cam_x, cam_y, cam_z])
    cam_fwd = -cam_pos.normalized()
    world_up = ti.Vector([0.0, 1.0, 0.0])
    
    return cam_pos, cam_fwd, world_up, fov

if __name__ == "__main__":
    main()