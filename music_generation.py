import glob
import numpy as np
from music21 import converter, instrument, note, chord
import random
from music21 import stream

# Function to extract notes and chords from MIDI files
def get_notes():
    notes = []
    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)
        notes_to_parse = None

        # Parse instrument parts
        try:  # File has instrument parts
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes

# Get the list of notes from all MIDI files
notes = get_notes()
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# Encode notes into integers and prepare sequences
def prepare_sequences(notes, n_vocab):
    sequence_length = 100

    # Map between notes and integers
    le = LabelEncoder()
    le.fit(notes)
    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([le.transform(list(sequence_in))])
        network_output.append(le.transform([sequence_out]))

    n_patterns = len(network_input)

    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)  # Normalize input
    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output, le

# Prepare sequences
n_vocab = len(set(notes))
network_input, network_output, le = prepare_sequences(notes, n_vocab)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.optimizers import Adam

def create_model(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model

# Create the model
model = create_model(network_input, n_vocab)
model.summary()
# Train the model
epochs = 200
batch_size = 64

model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size)

def generate_notes(model, network_input, le, n_vocab):
    start = random.randint(0, len(network_input) - 1)
    pattern = network_input[start]

    prediction_output = []

    # Generate 500 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = le.inverse_transform([index])[0]
        prediction_output.append(result)

        pattern = np.append(pattern, index)
        pattern = pattern[1:]

    return prediction_output

# Generate notes
generated_notes = generate_notes(model, network_input, le, n_vocab)

# Convert the output into a MIDI file
def create_midi(prediction_output, filename="output.mid"):
    offset = 0
    output_notes = []

    # Create note and chord objects based on the output
    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():  # It's a chord
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:  # It's a note
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=filename)

# Create a MIDI file from the generated notes
create_midi(generated_notes, "generated_music.mid")
