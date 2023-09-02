@REM argparse.ArgumentParser()
@REM     parser.add_argument("--input", type=str, required=True, help="Path to the npy file containing the annotations")
@REM     parser.add_argument("--output", type=str, required=True, help="Path to the output file for the train data")
@REM     parser.add_argument("--assert_int", action="store_true", help="Whether to assert that the values are integers")
@REM     args = parser.parse_args()
@REM     main(args)

python double_dataset.py ^
--input ..\datasets\cleaned\SBIC_annotations.npy ^
--output ..\datasets\cleaned\2xSBIC_annotations.npy ^
--assert_int