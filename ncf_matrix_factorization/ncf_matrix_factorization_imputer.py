import subprocess
import numpy as np
import tempfile
import os
import time

# imports from `kernel_matrix_factorization`
import sys
sys.path.append("../kernel_matrix_factorization")
import json
from imputer import Imputer, MatrixDataFrame, JSONstr, Tuple, pd
from npy_to_matrix_format import npy_to_matrix_format
from matrix_to_npy_format import matrix_to_npy_format

class NCFMatrixFactorizationImputer(Imputer):
    '''Impute the data using matrix factorization with a neural collaborative filtering model.
    '''
    def __init__(self, name: str, python_command: str, main_ours_search: str, batch_size: int=256, epochs: int=20):
        super().__init__(name)
        # Usually either "python" or "python3"
        self.python_command = python_command
        # location of the main_ours_search.py file
        self.main_ours_search = main_ours_search
        self.batch_size = batch_size
        self.epochs = epochs

    def impute(self, df: MatrixDataFrame) -> Tuple[pd.DataFrame, JSONstr]:
        '''Impute the data using matrix factorization with a neural collaborative filtering model.
        '''
        # create a tempfile to store the npy files
        with tempfile.TemporaryDirectory() as tmpdirname:
            annotations, texts = matrix_to_npy_format(df)
            input_annotations_path = os.path.abspath(os.path.join(tmpdirname, "input_annotations.npy"))
            np.save(input_annotations_path, annotations, allow_pickle=True)

            output_annotations_path = os.path.abspath(os.path.join(tmpdirname, "output_annotations"))

            # impute the entire df
            fold = -1
            # Construct the command
            cmd = f"{self.python_command} {self.main_ours_search} --input_path {input_annotations_path} --output_path {output_annotations_path} --fold {fold} --batch_size={self.batch_size} --epochs {self.epochs}"

            # Create a temporary file to store the output
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp_output_file:
                tmp_output_file_path = tmp_output_file.name

                # Run the command using subprocess with nohup and redirecting output to the tempfile
                full_command = f"CUDA_VISIBLE_DEVICES=0 nohup {cmd} > {tmp_output_file_path} 2>&1"
                print(f"About to run the following command (this may take a while):")
                print(full_command)
                process = subprocess.Popen(full_command, shell=True)

                # Poll the file to see if it has changed
                last_output_file_size = 0
                while process.poll() is None:
                    time.sleep(0.2) # Wait for a short period before polling again
                    current_output_file_size = os.path.getsize(tmp_output_file_path)
                    if current_output_file_size != last_output_file_size:
                        with open(tmp_output_file_path, "r") as updated_output_file:
                            updated_output_file.seek(last_output_file_size)
                            new_lines = updated_output_file.read()
                            print(new_lines.strip())

                        last_output_file_size = current_output_file_size

                # Read any remaining output after the process finishes
                with open(tmp_output_file_path, "r") as final_output_file:
                    final_output_file.seek(last_output_file_size)
                    remaining_output = final_output_file.read()
                    print(remaining_output.strip())

                # Remove the temporary output file
                os.remove(tmp_output_file_path)

            # if the process completed successfully
            if process.returncode == 0:
                # see main_ours_search.py for the naming convention
                loadable_output_annotations_path = output_annotations_path + "_" + str(fold) + ".npy"
                # load the output
                annotations = np.load(loadable_output_annotations_path, allow_pickle=True)
                texts = texts[-annotations.shape[0]:] # remove the unimputed examples at the start

                assert len(texts) == len(annotations), f"Error: len(texts)={len(texts)} != len(annotations)={len(annotations)}"

                # convert from npy to matrix format
                matrix_df = npy_to_matrix_format(annotations, texts)
                return matrix_df, json.dumps({})
            else:
                raise RuntimeError(f"Error: the process returned a non-zero exit code: {process.returncode}")