import moviepy.editor
import numpy as np
import tqdm


def get_chunk_decibels(waveform: np.ndarray):
    """
    Returns the decibels of the waveform.
    """
    if len(waveform.shape) == 2:  # Handle stereo data
        return (get_chunk_decibels(waveform[:, 0]) + get_chunk_decibels(waveform[:, 1])) / 2
    else:
        return 10 * np.log(np.sum(np.abs(waveform)) + 1e-8)


def filter(video):
    # https://github.com/Zulko/moviepy/issues/586
    if video.rotation == 90:
        video = video.resize(video.size[::-1])
        video.rotation = 0

    active_clips = []

    min_inactive_audio_cut = 0.5
    min_active_audio_db = 10

    i = 0
    frame = 0
    # If None, then the clip is not active. If not None, indicates when the activity began.
    active_start_frame = None
    # If not None, then we were active but then we suddenly dipped into inactivity.
    # We use this because we don't want to cut out very short dips into inactivity.
    inactive_start_frame = None
    for audio_chunk in tqdm.tqdm(video.audio.iter_chunks(chunk_duration=1/30), total=int(video.duration*30)):
        i += 1
        frame += len(audio_chunk)
        db = get_chunk_decibels(audio_chunk)

        if active_start_frame is not None:
            # Was active,
            if db < min_active_audio_db:
                # ...but not loud enough now

                # We skim over patches of silence less than min_inactive_audio_cut seconds long.
                if inactive_start_frame is None:
                    # Mark the beginning of silence
                    inactive_start_frame = frame

                time_since_silence_begin = (
                    frame - inactive_start_frame) / video.audio.fps

                if time_since_silence_begin >= min_inactive_audio_cut:
                    # Okay, cut it short now
                    # Go from the beginning of the activity to the beginning of the silence
                    start_time = active_start_frame / video.audio.fps
                    end_time = inactive_start_frame / video.audio.fps
                    active_clips.append(video.subclip(start_time, end_time))

                    # Reset the state
                    active_start_frame = None
                    inactive_start_frame = None
                else:
                    # Don't cut it short yet; there might be active audio before min_inactive_audio_cut seconds are up
                    pass
            else:
                # If it's currently active and we're above the dB threshold, don't do anything unusual
                # Just make sure we don't think there's silence
                if inactive_start_frame is not None:
                    inactive_start_frame = None
        else:
            # Was inactive,
            if db > min_active_audio_db:
                # ...but now it's loud enough
                active_start_frame = frame

    # If we are currently active, add the last clip
    if active_start_frame is not None:
        start_time = active_start_frame / video.audio.fps
        end_time = video.duration
        active_clips.append(video.subclip(start_time, end_time))

    # Concatenate the video clips
    # https://zulko.github.io/moviepy/getting_started/compositing.html
    composite = moviepy.editor.concatenate_videoclips(
        active_clips, method='compose')

    return composite


def save_video(video):
    # We need this workaround to get audio to render correctly
    # https://www.reddit.com/r/moviepy/comments/3uwuub/no_sound/
    video.write_videofile("IMG_4029_active.mp4", temp_audiofile="temp-audio.m4a",
                          remove_temp=True, codec="libx264", audio_codec="aac")


video = moviepy.editor.VideoFileClip('IMG_4029.MOV')
filtered = filter(video)
save_video(filtered)
filtered.close()
video.close()
