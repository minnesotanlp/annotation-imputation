from flask import Flask, render_template, request
import numpy as np
import random
import colorsys
import sys

app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)  # Add this line

dataset_name = 'SChem'
main_folder = f'data/{dataset_name}'

@app.route('/')
def index():
    examples = get_examples()
    return render_template('examples.html', examples=examples, dataset_name=dataset_name)

def get_examples():
    orig_annotations = np.round(np.load(f'{main_folder}/orig_annotations.npy', allow_pickle=True), 0)
    orig_annotations = orig_annotations.astype(int)
    imputed_kernel_annotations = np.round(np.load(f'{main_folder}/imputed_kernel_annotations.npy', allow_pickle=True), 0)
    imputed_kernel_annotations = imputed_kernel_annotations.astype(int)
    imputed_ncf_annotations = np.round(np.load(f'{main_folder}/imputed_ncf_annotations.npy', allow_pickle=True), 0)
    imputed_ncf_annotations = imputed_ncf_annotations.astype(int)
    orig_distribution = np.load(f'{main_folder}/orig_distribution.npy', allow_pickle=True)
    imputed_kernel_distribution = np.load(f'{main_folder}/imputed_kernel_distribution.npy', allow_pickle=True)
    imputed_ncf_distribution = np.load(f'{main_folder}/imputed_ncf_distribution.npy', allow_pickle=True)
    texts = np.load(f'{main_folder}/texts.npy', allow_pickle=True)
    kernel_kl = np.load(f'{main_folder}/kernel_kl.npy', allow_pickle=True)
    ncf_kl = np.load(f'{main_folder}/ncf_kl.npy', allow_pickle=True)

    # annotations and distribution should have the same length
    assert len(orig_annotations) == len(orig_distribution), f"{len(orig_annotations)} != {len(orig_distribution)}"
    assert len(imputed_kernel_annotations) == len(imputed_kernel_distribution), f"{len(imputed_kernel_annotations)} != {len(imputed_kernel_distribution)}"
    assert len(imputed_ncf_annotations) == len(imputed_ncf_distribution), f"{len(imputed_ncf_annotations)} != {len(imputed_ncf_distribution)}"

    # kernel should match original
    assert len(imputed_kernel_annotations) == len(orig_annotations), f"{len(imputed_kernel_annotations)} != {len(orig_annotations)}"

    # ncf should not be higher than original
    assert len(imputed_ncf_annotations) <= len(orig_annotations), f"{len(imputed_ncf_annotations)} > {len(orig_annotations)}"

    # if ncf is lower than original, remove the first few examples from the other two
    # this will usually be the case, because ncf does not impute its validation set
    start_index = 0
    if len(imputed_ncf_annotations) < len(orig_annotations):
        start_index = len(orig_annotations) - len(imputed_ncf_annotations)
        orig_annotations = orig_annotations[start_index:]
        orig_distribution = orig_distribution[start_index:]
        imputed_kernel_annotations = imputed_kernel_annotations[start_index:]
        imputed_kernel_distribution = imputed_kernel_distribution[start_index:]
        kernel_kl = kernel_kl[start_index:]
        texts = texts[start_index:]

    examples = []
    colors = generate_colors(len(orig_distribution[0]))
    # black = '#000000'

    for idx in range(len(texts)):
        assert len(orig_annotations[idx]) == len(imputed_kernel_annotations[idx]) == len(imputed_ncf_annotations[idx]), f"{len(orig_annotations[idx])} != {len(imputed_kernel_annotations[idx])} != {len(imputed_ncf_annotations[idx])}"
        match_indexes = [i for i in range(len(orig_annotations[idx])) if orig_annotations[idx][i] != -1]

        orig_matching_annotations = [int(orig_annotations[idx][i]) for i in match_indexes]
        orig_colors = [colors[annotation] for annotation in orig_matching_annotations]

        imputed_matching_kernel_annotations = [imputed_kernel_annotations[idx][i] for i in match_indexes]
        imputed_kernel_colors = [colors[annotation] for annotation in imputed_matching_kernel_annotations]

        imputed_matching_ncf_annotations = [imputed_ncf_annotations[idx][i] for i in match_indexes]
        imputed_ncf_colors = [colors[annotation] for annotation in imputed_matching_ncf_annotations]

        example = {
            'index': idx + start_index,
            'text': texts[idx],
            'orig_annotations': orig_matching_annotations,
            'orig_distribution': np.round(orig_distribution[idx], 3),
            'orig_colors': orig_colors,
            'imputed_kernel_annotations': imputed_matching_kernel_annotations,
            'imputed_kernel_distribution': np.round(imputed_kernel_distribution[idx], 3),
            'imputed_kernel_colors': imputed_kernel_colors,
            'kernel_kl': round(kernel_kl[idx], 3),
            'imputed_ncf_annotations': imputed_matching_ncf_annotations,
            'imputed_ncf_distribution': np.round(imputed_ncf_distribution[idx], 3),
            'imputed_ncf_colors': imputed_ncf_colors,
            'ncf_kl': round(ncf_kl[idx], 3),
            'colors': colors
        }
        examples.append(example)

    return examples

def generate_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# def generate_colors(n):
#     return [generate_color() for _ in range(n)]

def generate_colors(n):
    hsv_colors = [(i / n, 1, 1) for i in range(n)]
    rgb_colors = [tuple(int(255 * x) for x in colorsys.hsv_to_rgb(*c)) for c in hsv_colors]
    return ['#%02x%02x%02x' % c for c in rgb_colors]

if __name__ == '__main__':
    app.run(debug=True)