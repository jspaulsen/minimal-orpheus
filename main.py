import io
import time

import wave # This is actually built-in, wild.

from src.tts.orpheus.generator import Orpheus, OrpheusModels


random_sentences = [
    "The cat stared at the toaster like it was plotting something.",
    "I never expected to find a violin in the fridge.",
    "She danced through the puddles without a care in the world.",
    "Jupiter is mostly made of gas, yet it holds dozens of moons.",
    "He wore mismatched socks to the job interview on purpose.",
    "A bird sang outside my window all night long.",
    "The robots competed in a cooking competition on Mars.",
    "She accidentally invented a new color while painting.",
    "Time machines should come with a manual, he thought.",
    "The stars blinked like tiny lighthouses in the dark sky.",
    "Bananas shouldn't be used as walkie-talkies, but here we are.",
    "He whispered secrets to the plants every morning.",
    "Clouds shaped like dinosaurs filled the afternoon sky.",
    "A fish once told me its dreams through interpretive dance.",
    "She wrote a novel entirely in emoji.",
    "The microwave beeped a secret Morse code message.",
    "He wore a cape to the grocery store just for fun.",
    "Snowflakes danced in the wind like tiny ballerinas.",
    "A penguin waddled into the courtroom unannounced.",
    "The elevator music was suspiciously funky today.",
    "He bought a rubber chicken instead of milk by mistake.",
    "The haunted toaster only made ghost-shaped waffles.",
    "Her shoes squeaked with every confident step.",
    "The sun blinked once, and then everything changed.",
    "I found a treasure map drawn in jelly on toast.",
    "The vending machine gave me a potato instead of chips.",
    "She spoke fluent whale after just two weeks of practice.",
    "Rain fell in reverse during the strangest storm ever.",
    "My reflection winked at me before I did.",
    "He painted his car like a giant watermelon.",
    "The moon hummed a lullaby to the sleeping earth.",
    "She collected buttons from alternate dimensions.",
    "A snail beat me at chess and refused a rematch.",
    "The fridge demanded tribute before opening.",
    "He wore sunglasses indoors to hide his superhero identity.",
    "Music played from nowhere and everywhere at once.",
    "I saw a cloud shaped exactly like a rubber duck.",
    "She trained her cactus to dance salsa.",
    "The dog wrote a poem and buried it in the yard.",
    "Books whispered forgotten tales when the lights went out.",
    "The mirror showed a future I didn't recognize.",
    "He turned his garden into a miniature theme park.",
    "She chased a butterfly into another universe.",
    "Coffee tasted like victory this morning.",
    "The train to Nowhere arrived five minutes early.",
    "He built a pillow fort that could withstand earthquakes.",
    "The typewriter typed stories on its own every night.",
    "She found a universe inside a snow globe.",
    "The squirrel held a tiny protest in the backyard.",
    "He wore a tie covered in miniature ducks for luck."
]


def generate_audio_rtf(orpheus: Orpheus, sentence: str):
    audio_gen = orpheus.generate_audio(
        sentence,
        "Tara",
    )

    now = time.time()
    frames = []

    for c in audio_gen:
        frames.append(c)

    run =  time.time() - now

    # Need to clean this up
    audio_buffer = io.BytesIO()

    # Need to concatenate and save the audio chunks
    wf = wave.open(audio_buffer, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)

    for frame in frames:
        wf.writeframes(frame)

    num_frames = wf.getnframes()
    frame_rate = wf.getframerate()

    wf.close()

    duration = num_frames / frame_rate
    rtf = run / duration

    # print(f"Duration: {duration:.2f}s, RTF: {rtf:.2f}, TTG: {run:.2f}s")
    return rtf


def main():
    orpheus = Orpheus(
        OrpheusModels.MungertQ4_K_M,
        gpu=1,
    )

    runs = []

    for i, sentence in enumerate(random_sentences[:10]):
        run = generate_audio_rtf(orpheus, sentence)

        if i > 0:
            runs.append(run)

    print(f"Average RTF: {sum(runs) / len(runs):.2f}")
    print(f"Max RTF: {max(runs):.2f}")
    print(f"Min RTF: {min(runs):.2f}")


if __name__ == '__main__':
    main()
