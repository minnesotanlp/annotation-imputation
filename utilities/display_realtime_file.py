import os
import time
import argparse

def display_realtime_file(file_location: str):
    '''Displays the contents of a file in real time. You will need to CTRL+C to stop the program.'''
    print(f"BEGINNING REAL-TIME DISPLAY OF FILE: {file_location}")
    last_file_size = 0
    while True:
        current_file_size = os.path.getsize(file_location)
        if current_file_size != last_file_size:
            with open(file_location, "r") as updated_file:
                updated_file.seek(last_file_size)
                new_data = updated_file.read()
                print(new_data, end='', flush=True)  # Print without newline and flush the output buffer
            last_file_size = current_file_size
        time.sleep(0.1)  # Wait for a short period before polling again

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display the contents of a file in real time.')
    parser.add_argument('file_location', type=str, default='display_realtime_file.py', help='The location of the file to display.')
    args = parser.parse_args()
    display_realtime_file(args.file_location)
    # then add a comment to the file and save it
    # remember that it's only going to print the data that increases the size of the file, so if you delete a line and then add a new one, it likely won't print anything