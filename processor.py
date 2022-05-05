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

    min_inactive_audio_cut = 0.05
    min_active_audio_db = 10

    i = 0
    frame = 0
    # if None, then the clip is not active. if not none, indicates when the activity began.
    last_active_audio_start_frame = None
    for audio_chunk in tqdm.tqdm(video.audio.iter_chunks(chunk_duration=1/30), total=int(video.duration*30)):
        i += 1
        frame += len(audio_chunk)
        db = get_chunk_decibels(audio_chunk)

        if last_active_audio_start_frame is not None:
            # Was active,
            if db < min_active_audio_db:
                # ...but not loud enough now
                start_time = last_active_audio_start_frame / video.audio.fps
                end_time = frame / video.audio.fps
                active_clips.append(video.subclip(start_time, end_time))
                # Wait until min_inactive_audio_cut seconds have passed to reset last active audio start frame
                if (frame - last_active_audio_start_frame) / video.audio.fps > min_inactive_audio_cut:
                    last_active_audio_start_frame = None

            # If it's currently active and we're above the dB threshold, don't do anything unusual
        else:
            # Was inactive,
            if db > min_active_audio_db:
                # ...but now it's loud enough
                last_active_audio_start_frame = frame

    # https://zulko.github.io/moviepy/getting_started/compositing.html
    composite = moviepy.editor.concatenate_videoclips(
        active_clips, method='compose')

    # https://www.reddit.com/r/moviepy/comments/3uwuub/no_sound/
    composite.write_videofile("IMG_4029_active.mp4", temp_audiofile="temp-audio.m4a",
                              remove_temp=True, codec="libx264", audio_codec="aac")
    composite.close()

    video.close()


video = moviepy.editor.VideoFileClip('IMG_4029.MOV')
filter(video)
