import ast
import pandas as pd
import plotly.graph_objects as go
from tqdm.auto import tqdm
tqdm.pandas()

df_human = pd.read_csv('worker_human.csv')
df_machine = pd.read_csv('worker_machine.csv')
dfs = [df_human, df_machine]

# convert metaphor count into percentage
for df in dfs:
    df['metaphor'] = df['metaphor'].progress_apply(lambda x: ast.literal_eval(x))
    df['metaphor count'] = df['metaphor'].progress_apply(lambda dict_list: sum([1 for d in dict_list if d['entity'] == 'LABEL_1'])/len(dict_list))

# convert attribute scores into binary labels
for df in dfs:
    df['irony label'] = df['irony'].apply(lambda x: 1 if x >=0.5 else 0)
    df['formality label'] = df['formality'].apply(lambda x: 1 if x >=0.5 else 0)
    df['toxicity label'] = df['toxicity'].apply(lambda x: 1 if x >=0.5 else 0)
    for emotion in ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']:
        df[f'{emotion} label'] = df[emotion].apply(lambda x: 1 if x >=0.5 else 0)

# find 'flipped' labels
human_df, machine_df = dfs
for label in ['irony label', 'formality label', 'anger label', 'disgust label',
              'fear label', 'joy label', 'neutral label', 'sadness label',
              'surprise label', 'toxicity label', 'metaphor count']:
    human_df['human_id'] = human_df['id']
    merged = pd.merge(machine_df, human_df, how='inner', on='human_id')
    machine_df[f'flipped {label} 0m1h'] = (merged[f'{label}_x'] == 0) & (merged[f'{label}_y'] == 1)
    machine_df[f'flipped {label} 1m0h'] = (merged[f'{label}_x'] == 1) & (merged[f'{label}_y'] == 0)
    if label == 'metaphor count':
        machine_df[f'flipped {label} 0m1h'] = merged.apply(lambda row: (row[f'{label}_y'] - row[f'{label}_x']) > 0.15, axis=1)
        machine_df[f'flipped {label} 1m0h'] = merged.apply(lambda row: (row[f'{label}_x'] - row[f'{label}_y']) > 0.15, axis=1)


### Create Figure
categories = ['social_support', 'similarity_identity', 'respect', 'knowledge', 'power', 'trust', 'fun', 'conflict', 'neutral', 'positive', 'negative']
label_cats = ['social support', 'similar identity', 'respect', 'knowledge', 'power', 'trust', 'fun', 'conflict', 'neutral', 'positive', 'negative', 'social support']
emotions = ['joy', 'neutral', 'sadness']

fig = go.Figure()
color_scheme = ['#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
label_vertex_map = {'neutral': 7, 'joy': 9, 'sadness': 10}

for i, label_col in enumerate([c for c in machine_df.columns if '1m0h' in c]):
    has_emotion = False
    for emotion in emotions:
        if emotion in label_col:
            has_emotion = True
    if not has_emotion:
        continue
    r = []
    for cat in categories:
        subdf = machine_df.loc[machine_df['label'] == cat]
        r.append(subdf[label_col].mean())
    r.append(r[0])
    l = label_col.split()[1]
    fig.add_trace(go.Scatterpolar(
        mode='lines+text',
        r=r,
        theta=label_cats,
        name='',
        text=['' if label_vertex_map[l] != ind else l for ind in range(len(r))],
        textfont=dict(size=16, color=color_scheme[len(fig.data)]),
        textposition='bottom right',
        line=dict(color=color_scheme[len(fig.data)])
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 0.5],
            tickmode='array',
            tickvals = [0, 0.25, 0.5],
            ticktext = ['', '25%', '50%'],
            tickfont = dict(size=13),
            ticks = 'outside',
            ticklen = 1,
            linewidth=0,
            color='#444',
        )
    ),
    showlegend=False,
    font=dict(color='black', size=18),
)

fig.show()
