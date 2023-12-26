from scipy.stats import wasserstein_distance
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

MODEL_SIZE = {
    'ada' : '350M',
    'babbage': '1.3B',
    'curie' : '6.7B',
    'davinci': '175B',
    'text-ada-001': '350M',
    'text-curie-001' : '6.7B',
    'text-davinci-001': '175B',
}

def calculate_majority(lst): # length of list is always 5 as 5 annotators
    """
    Calculate the majority value(s) from the list.

    :param lst: List of integers with values in the range [0, 1, 2, 3, 4] and possibly -1.
    :return: List of majority values. If no majority exists or all values are different, returns None.
    """
    # Remove -1 values from the list
    valid_values = [x for x in lst if x != -1]

    # If the list is empty after removing -1 values, return None
    if not valid_values:
        return None

    # Create a dictionary to count occurrences of each value
    count_dict = {}
    for value in valid_values:
        count_dict[value] = count_dict.get(value, 0) + 1

    # Find the maximum occurrence
    max_count = max(count_dict.values())

    # If the maximum occurrence is 1, then all values are unique
    if max_count == 1:
        return None

    # Extract all values that have the maximum occurrence
    majority_values = [key for key, value in count_dict.items() if value == max_count]

    return majority_values

def calculate_minority(lst):
    """
    Calculate the minority value(s) from the list.

    :param lst: List of integers with values in the range [0, 1, 2, 3, 4] and possibly -1.
    :return: List of minority values. If no minority exists or all values have equal occurrences, returns None.
    """
    # Remove -1 values from the list
    valid_values = [x for x in lst if x != -1]

    # If the list is empty after removing -1 values, return None
    if not valid_values:
        return None

    # Create a dictionary to count occurrences of each value
    count_dict = {}
    for value in valid_values:
        count_dict[value] = count_dict.get(value, 0) + 1

    # Find the minimum occurrence
    min_count = min(count_dict.values())

    # If the minimum occurrence is equal to the length of valid values divided by the number of unique valid values, then all values have equal occurrences
    if min_count == len(valid_values) / len(count_dict):
        return None

    # Extract all values that have the minimum occurrence
    minority_values = [key for key, value in count_dict.items() if value == min_count]

    return minority_values


def calculate_wasserstein_distance(row):

    categories = list(set(row['machine_maj_agg'] + row['human_annots']))

    machine_counts = np.array([row['machine_maj_agg'].count(cat) for cat in categories]) + 1  # Laplace smoothing
    human_counts = np.array([row['human_annots'].count(cat) for cat in categories]) + 1  # Laplace smoothing

    # Convert counts to probabilities
    machine_probs = machine_counts / machine_counts.sum()
    human_probs = human_counts / human_counts.sum()

    # Compute 1-Wasserstein Distance
    wasserstein_dist = wasserstein_distance(machine_probs, human_probs)
    return wasserstein_dist

# Function to convert string representation of list to actual list and convert nans to -1
def str_to_list(s):
    elements = s.strip('[]').replace("'", "").split()
    # Converting each element to float; if 'nan', then convert to -1
    result = []
    for e in elements:
        if e.lower() =='nan':
            result.append(-1)
        else:
            float_value = float(e)
            result.append(float_value)  # try to convert to float

    return result

# Function to convert out of range predictions to -1
# this defines that if a prediction is encountered > value value it should be -1
range_dict =  {'SBIC': 2, 'ghc':1, 'Sentiment': 4, 'SChem5Labels' : 4}
# Function to be applied to each row
def process_row(row):
    dataset_name = row['dataset_name']
    model_annots = row['model_annots']

    # Get threshold value for dataset_name
    threshold = range_dict.get(dataset_name, -1)  # Default to -1 if dataset_name not found in range_dict

    # Process model_annots list
    new_model_annots = [-1 if val > threshold else val for val in model_annots]

    # Update the row
    row['model_annots'] = new_model_annots
    return row

# Apply function to DataFrame
# df = df.apply(process_row, axis=1)

# Assuming df is your dataframe
# agg_df['Wasserstein_Distance'] = agg_df.apply(lambda row: calculate_wasserstein_distance(row), axis=1) 

def get_human_labels(fname):

    data = np.load(fname, allow_pickle=True)
    return [ int(j) for i,j in data]

# print(get_human_labels('./human-labels/SBIC_10_agg_test.npy'))

def get_gpt_labels(fname):

    data = open(fname)

    gpt_list = []

    for row in data: 
        rjson = json.loads(row)
        out = rjson['generation']

        f = out.split(' ')[0]
        ok = False

        for v in f:
            if v <= '9' and v >= '0':
                gpt_list.append(int(v))
                ok = True
                break
                
        if not ok:
            gpt_list.append(-1)

    return gpt_list

# print(get_gpt_labels('./ada/out_ada_sbic.json'))

def get_matched_samples(hlabels,mlabels):

    print(len(hlabels), len(mlabels))

    assert len(hlabels) == len(mlabels)

    tot = 0
    cnt_matched = 0

    for i in range(len(hlabels)):
        if mlabels[i] != -1:
            tot += 1
            if hlabels[i] == mlabels[i]:
                cnt_matched += 1

    return cnt_matched/tot

# print(get_matched_samples(get_human_labels('./human-labels/SBIC_10_agg_test.npy'),get_gpt_labels('./ada/out_ada_sbic.json')))

def plot_models(models, dataset_name, plot_type='sep'):

    base_model_path = './'
    base_human_path = './human-labels'

    ratio_matched_samples = []
    ck_scores = []

    for model in models: 
        model_path = base_model_path + model + '/out_{0}_{1}.json'.format(model, dataset_name.lower())
        human_path = base_human_path + '/{0}_10_agg_test.npy'.format(dataset_name)

        ratio_matched_samples.append(get_matched_samples(get_human_labels(human_path),get_gpt_labels(model_path)))
        ck_scores.append(cohen_kappa_score(get_human_labels(human_path),get_gpt_labels(model_path)))

    if plot_type == 'sep':
        plt.plot(models,ratio_matched_samples,marker='',linewidth=2)
    elif plot_type == 'all':
        plt.plot([ MODEL_SIZE[model] for model in models ], ratio_matched_samples,marker='',linewidth=2)
    # plt.savefig('plot_{0}.png'.format(dataset_name))

    print(ratio_matched_samples)
    print(ck_scores)

# plt.xlabel('Model Size')
# plt.ylabel('Matched Score')
# plot_models(['ada','babbage','curie','davinci'],'SBIC','sep')
# plot_models(['ada','babbage','curie','davinci'],'ghc','sep')
# plot_models(['ada','babbage','curie','davinci'],'SChem5Labels','sep')
# plot_models(['ada','babbage','curie','davinci'],'Sentiment','sep')

# plot_models(['text-ada-001','text-curie-001','text-davinci-001'],'SBIC','sep')
# plot_models(['text-ada-001','text-curie-001','text-davinci-001'],'ghc','sep')
# plot_models(['text-ada-001','text-curie-001','text-davinci-001'],'SChem5Labels','sep')
# plot_models(['text-ada-001','text-curie-001','text-davinci-001'],'Sentiment','sep')

# plot_models(['ada','babbage','curie','davinci'],'SBIC','all')
# plot_models(['ada','babbage','curie','davinci'],'ghc','all')
# plot_models(['ada','babbage','curie','davinci'],'SChem5Labels','all')
# plot_models(['ada','babbage','curie','davinci'],'Sentiment','all')

# plot_models(['text-ada-001','text-curie-001','text-davinci-001'],'SBIC','all')
# plot_models(['text-ada-001','text-curie-001','text-davinci-001'],'ghc','all')
# plot_models(['text-ada-001','text-curie-001','text-davinci-001'],'SChem5Labels','all')
# plot_models(['text-ada-001','text-curie-001','text-davinci-001'],'Sentiment','all')

# plt.legend(['sbic','ghc','schem5labels','sentiment'])
# plt.legend(['sbic','ghc','schem5labels','sentiment','sbic-instruct','ghc-instruct','schem5labels-instruct','sentiment-instruct'])

# plt.savefig('plot_all_instruct.png')

# plt.tight_layout()
# plt.subplots(layout="constrained")
# plt.subplot(221)
# plt.xlabel('Model Size')
# plt.ylabel('Matched Score')
# plot_models(['ada','babbage','curie','davinci'],'SBIC','all')
# plot_models(['text-ada-001','text-curie-001','text-davinci-001'],'SBIC','all')
# plt.legend(['sbic','sbic-instruct'])
# plt.subplot(222)
# plt.xlabel('Model Size')
# plt.ylabel('Matched Score')
# plot_models(['ada','babbage','curie','davinci'],'ghc','all')
# plot_models(['text-ada-001','text-curie-001','text-davinci-001'],'ghc','all')
# plt.legend(['ghc','ghc-instruct'])
# plt.subplot(223)
# plt.xlabel('Model Size')
# plt.ylabel('Matched Score')
# plot_models(['ada','babbage','curie','davinci'],'SChem5Labels','all')
# plot_models(['text-ada-001','text-curie-001','text-davinci-001'],'SChem5Labels','all')
# plt.legend(['schem5labels','schem5labels-instruct'])
# plt.subplot(224)
# plt.xlabel('Model Size')
# plt.ylabel('Matched Score')
# plot_models(['ada','babbage','curie','davinci'],'Sentiment','all')
# plot_models(['text-ada-001','text-curie-001','text-davinci-001'],'Sentiment','all')
# plt.legend(['sentiment','sentiment-instruct'])

# plt.legend(['sbic','sbic-instruct'],['ghc','ghc-instruct'],['schem5labels','schem5labels-instruct'],['sentiment','sentiment-instruct'])

# plt.savefig('plot_diff_data.png')

# plt.plot('')
plot_models(['gpt-3.5-turbo'],'Sentiment')
plot_models(['gpt-3.5-turbo'],'SChem5Labels')
plot_models(['gpt-3.5-turbo'],'ghc')
plot_models(['gpt-3.5-turbo'],'SBIC')



