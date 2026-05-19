"""
app.py — Director Fingerprint
Streamlit app — deploy on share.streamlit.io

Architecture:
- Film database (film_database.json): pre-extracted features for ~200 films
- Live fallback: OpenSubtitles API for films not in database
- Classifier: Random Forest trained on 25 films across 5 directors
"""

import streamlit as st
import requests
import re
import json
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Director Fingerprint",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 800px; margin: 0 auto; }
    .stTextInput input { font-size: 18px; padding: 12px; }
    .metric-container { background: #f8f8f6; border-radius: 12px; padding: 16px; }
    .winner-name { font-size: 36px; font-weight: 600; letter-spacing: -1px; }
    .section-label { font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
                     text-transform: uppercase; color: #888; margin-bottom: 8px; }
    div[data-testid="stProgress"] > div { background: #f0f0ee; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_database():
    if Path('film_database.json').exists():
        with open('film_database.json') as f:
            return json.load(f)
    return {}

model_data = load_model()
model      = model_data['model']
scaler     = model_data['scaler']
FEATURES   = model_data['features']
film_db    = load_database()

# Build title → imdb_id lookup for fast search
TITLE_INDEX = {}
for imdb_id, film in film_db.items():
    key = film['title'].lower().strip()
    TITLE_INDEX[key] = imdb_id
    # Also index without articles
    for article in ['the ', 'a ', 'an ']:
        if key.startswith(article):
            TITLE_INDEX[key[len(article):]] = imdb_id

# ── OpenSubtitles API (fallback for films not in database) ────────────────────
def get_api_key():
    try:
        return st.secrets["OPENSUBTITLES_API_KEY"]
    except:
        return None

API_KEY  = get_api_key()
BASE_URL = "https://api.opensubtitles.com/api/v1"

@st.cache_data(ttl=3600)
def fetch_and_extract(movie_name: str):
    """Fetch subtitle from API and extract features. Cached for 1 hour."""
    if not API_KEY:
        return None, "No API key configured"

    headers = {"Api-Key": API_KEY, "User-Agent": "DirectorFingerprint/1.0"}

    # Search
    r = requests.get(f"{BASE_URL}/subtitles",
        params={"query": movie_name, "languages": "en", "order_by": "download_count"},
        headers=headers, timeout=10)

    if r.status_code != 200:
        return None, f"Search failed ({r.status_code})"

    results = r.json().get("data", [])
    if not results:
        return None, f"No English subtitles found for '{movie_name}'"

    file_id = results[0]["attributes"]["files"][0]["file_id"]

    # Get download link
    r2 = requests.post(f"{BASE_URL}/download",
        json={"file_id": file_id},
        headers={**headers, "Content-Type": "application/json"},
        timeout=10)

    if r2.status_code != 200:
        return None, f"Download request failed ({r2.status_code})"

    dl_url = r2.json().get("link")
    if not dl_url:
        return None, "No download link returned"

    # Download
    r3 = requests.get(dl_url, timeout=15)
    if r3.status_code != 200:
        return None, "Subtitle download failed"

    features = extract_features(r3.text)
    if not features:
        return None, "Could not parse subtitle file"

    return features, None

# ── Feature extraction ────────────────────────────────────────────────────────
def parse_srt(content):
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    content = re.sub(r'<[^>]+>', '', content)
    content = re.sub(r'\{[^}]+\}', '', content)
    blocks  = re.split(r'\n\n+', content.strip())
    subs    = []
    for block in blocks:
        lines   = block.strip().split('\n')
        ts_line = next((l for l in lines if '-->' in l), None)
        if not ts_line: continue
        ts_idx  = lines.index(ts_line)
        try:
            parts = ts_line.split('-->')
            start = ts_to_sec(parts[0].strip())
            end   = ts_to_sec(parts[1].strip().split()[0])
        except: continue
        text = ' '.join(lines[ts_idx+1:]).strip()
        text = re.sub(r'\s+', ' ', text)
        if len(text) < 2: continue
        subs.append({'start': start, 'end': end, 'text': text})
    return subs

def ts_to_sec(ts):
    h, m, rest = ts.strip().split(':')
    s, ms = rest.replace(',', '.').split('.')
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

def variance(lst):
    if len(lst) < 2: return 0.0
    m = sum(lst)/len(lst)
    return sum((x-m)**2 for x in lst)/len(lst)

def extract_features(srt_content):
    subs = parse_srt(srt_content)
    if len(subs) < 20: return None

    film_dur_raw = subs[-1]['end']
    subs = [s for s in subs if s['start'] > 180 and s['end'] < film_dur_raw - 180]
    if len(subs) < 20: return None

    film_dur  = subs[-1]['end'] - subs[0]['start']
    durations = [s['end'] - s['start'] for s in subs]
    gaps = [subs[i]['start'] - subs[i-1]['end']
            for i in range(1, len(subs))
            if subs[i]['start'] - subs[i-1]['end'] > 0]

    texts      = [s['text'] for s in subs]
    words      = re.findall(r"[a-zA-Z']+", ' '.join(texts).lower())
    vocab_rich = len(set(words)) / len(words) if words else 0
    line_lens  = [len(re.findall(r"[a-zA-Z']+", t)) for t in texts]

    bursts = []; cur = 1
    for i in range(1, len(subs)):
        if subs[i]['start'] - subs[i-1]['end'] < 1.5: cur += 1
        else: bursts.append(cur); cur = 1
    bursts.append(cur)

    act_dens = []
    for act in range(10):
        t0 = subs[0]['start'] + act * film_dur/10
        t1 = t0 + film_dur/10
        act_dens.append(sum(1 for s in subs if t0 <= s['start'] < t1))

    waveform = []
    for i in range(20):
        t0 = subs[0]['start'] + i * film_dur/20
        t1 = t0 + film_dur/20
        waveform.append(sum(1 for s in subs if t0 <= s['start'] < t1))

    return {
        'silence_ratio':     round(1 - sum(durations)/film_dur, 4),
        'mean_gap_sec':      round(sum(gaps)/len(gaps) if gaps else 0, 3),
        'long_silence_rate': round(sum(1 for g in gaps if g > 5)/len(gaps) if gaps else 0, 4),
        'silence_intensity': round(sum(1 for g in gaps if g > 15)/len(gaps) if gaps else 0, 4),
        'vocab_richness':    round(vocab_rich, 4),
        'mean_line_words':   round(sum(line_lens)/len(line_lens) if line_lens else 0, 3),
        'question_ratio':    round(sum(1 for t in texts if '?' in t)/len(texts), 4),
        'exclaim_ratio':     round(sum(1 for t in texts if '!' in t)/len(texts), 4),
        'mean_burst_len':    round(sum(bursts)/len(bursts), 3),
        'long_burst_rate':   round(sum(1 for b in bursts if b >= 5)/len(bursts), 4),
        'subs_per_minute':   round(len(subs)/(film_dur/60), 3),
        'pacing_variance':   round(variance(act_dens), 3),
        'pacing_front_heavy':round(sum(act_dens[:5])/(sum(act_dens[5:])+1), 3),
        'film_duration_min': round(film_dur/60, 1),
        'n_subtitles':       len(subs),
        'waveform':          waveform,
    }

# ── Classification ────────────────────────────────────────────────────────────
def classify(features):
    x = np.array([features[f] for f in FEATURES]).reshape(1, -1)
    x_sc = scaler.transform(x)
    proba = model.predict_proba(x_sc)[0]
    return {d: float(p) for d, p in zip(model.classes_, proba)}

# ── Cinematic traits ──────────────────────────────────────────────────────────
COLORS = {
    'Kurosawa':'#2D6FA3','Kubrick':'#5A5A5A','Miyazaki':'#2D8C65',
    'Tarantino':'#B84C2A','Kashyap':'#6355B8','Scorsese':'#B8962A',
}
DEFAULT_COLOR = '#2D6FA3'

DIRECTOR_AVGS = {
    'Kurosawa': {'Economy':0.80,'Intensity':0.28,'Interrogation':0.72,'Emotion':0.38},
    'Kubrick':  {'Economy':0.74,'Intensity':0.62,'Interrogation':0.61,'Emotion':0.22},
    'Miyazaki': {'Economy':0.55,'Intensity':0.52,'Interrogation':0.46,'Emotion':0.50},
    'Tarantino':{'Economy':0.22,'Intensity':0.88,'Interrogation':0.74,'Emotion':0.88},
    'Kashyap':  {'Economy':0.58,'Intensity':0.56,'Interrogation':0.48,'Emotion':0.58},
}

TRAIT_DESCS = {
    'Economy': {
        'low': 'Words flood the film. The same phrases surface again and again — deliberately.',
        'mid': 'Balanced. Words serve the scene without excess.',
        'high': 'Spare. Each word earns its place. Silence does the rest.',
    },
    'Intensity': {
        'low': 'Measured. Dialogue arrives as pronouncements, not conversation.',
        'mid': 'Fluid. Exchanges build and pause naturally.',
        'high': 'Rapid and relentless. Long volleys, momentum above all.',
    },
    'Interrogation': {
        'low': 'Declarative. Characters state rather than ask.',
        'mid': 'Mixed. Questions arise at moments of conflict.',
        'high': 'Everything is a question. Uncertainty drives every scene.',
    },
    'Emotion': {
        'low': 'Flat and controlled. Feeling is implied, never announced.',
        'mid': 'Present but restrained. Emotion surfaces without theatricality.',
        'high': 'Loud and theatrical. Emotion is worn externally, loudly.',
    }
}

def compute_traits(features):
    return {
        'Economy':       min(1.0, features['vocab_richness'] / 0.28),
        'Intensity':     min(1.0, (features['mean_burst_len'] - 1.5) / 5.5),
        'Interrogation': min(1.0, features['question_ratio'] / 0.25),
        'Emotion':       min(1.0, features['exclaim_ratio'] / 0.30),
    }

def trait_desc(trait, val):
    d = TRAIT_DESCS[trait]
    if val < 0.33: return d['low']
    elif val < 0.67: return d['mid']
    else: return d['high']

TRAIT_LABELS = {
    'Economy':       'Economy of language',
    'Intensity':     'Intensity of exchange',
    'Interrogation': 'Interrogation',
    'Emotion':       'Emotional register',
}

# ── Lookup film ───────────────────────────────────────────────────────────────
def lookup_film(query):
    """Returns (features_dict, source_str, error_str)"""
    q = query.lower().strip()

    # Exact match in database
    if q in TITLE_INDEX:
        imdb_id = TITLE_INDEX[q]
        film    = film_db[imdb_id]
        return film, f"from database · {film.get('film_duration_min',0):.0f} min", None

    # Partial match
    for title_key, imdb_id in TITLE_INDEX.items():
        if q in title_key or title_key in q:
            film = film_db[imdb_id]
            return film, f"from database · {film.get('film_duration_min',0):.0f} min", None

    # Fallback: live API
    features, err = fetch_and_extract(query)
    if err:
        return None, None, err
    return features, "fetched live via OpenSubtitles", None

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("## Director fingerprint")
st.markdown("Type any film title. We read how it breathes.")
st.markdown("")

query = st.text_input(
    label="film_search",
    placeholder="e.g.  Parasite   ·   The Godfather   ·   Spirited Away   ·   Dune",
    label_visibility="collapsed"
)

# Show database size
n_db = len(film_db)
if n_db > 0:
    st.caption(f"{n_db} films in database · instant results for most searches")

if query and len(query.strip()) > 1:
    with st.spinner(""):
        features, source, error = lookup_film(query.strip())

    if error:
        st.error(f"Couldn't find '{query}'. Try adding the year — e.g. 'Parasite 2019'")
    else:
        proba  = classify(features)
        traits = compute_traits(features)

        sorted_proba = sorted(proba.items(), key=lambda x: -x[1])
        winner, wp   = sorted_proba[0]
        runner, rp   = sorted_proba[1]
        color        = COLORS.get(winner, DEFAULT_COLOR)

        # ── Header ───────────────────────────────────────────────────────────
        st.markdown("---")
        title_display = features.get('title', query.title())
        year_display  = features.get('year', '')

        col_main, col_source = st.columns([3,1])
        with col_main:
            st.markdown(f"### {title_display}{' (' + str(year_display) + ')' if year_display else ''}")
            st.caption(source)

        st.markdown(f"<div class='winner-name' style='color:{color}'>{winner}</div>", unsafe_allow_html=True)
        st.caption(f"{round(wp*100)}% match")

        # ── Probability bar ───────────────────────────────────────────────────
        st.markdown("")
        prob_cols = st.columns(len(sorted_proba))
        for i, (d, p) in enumerate(sorted_proba):
            with prob_cols[i]:
                c = COLORS.get(d, DEFAULT_COLOR)
                st.markdown(f"<div style='font-size:12px;color:{c};font-weight:600'>{d}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:18px;font-weight:500'>{round(p*100)}%</div>", unsafe_allow_html=True)
                st.progress(p)

        st.markdown("---")

        # ── Dialogue waveform ─────────────────────────────────────────────────
        st.markdown("<div class='section-label'>Dialogue rhythm — where this film breathes and where it rushes</div>", unsafe_allow_html=True)

        waveform = features.get('waveform', [5]*20)
        max_w    = max(waveform) if waveform else 1

        fig_wave = go.Figure(go.Bar(
            y=waveform,
            marker_color=color,
            marker_opacity=[0.2 + 0.8*(v/max_w) for v in waveform],
            hovertemplate='%{y} subtitle blocks<extra></extra>',
        ))
        fig_wave.update_layout(
            height=100,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
        )
        st.plotly_chart(fig_wave, use_container_width=True, config={'displayModeBar': False})

        act_labels = ['Opening', '', 'Act 1', '', 'Midpoint', '', 'Act 2', '', 'Climax', '', 'Close']
        label_cols = st.columns(len(waveform))
        for i, col in enumerate(label_cols):
            with col:
                idx = round(i * 10 / (len(waveform)-1))
                if idx < len(act_labels) and act_labels[idx]:
                    st.markdown(f"<div style='font-size:10px;color:#bbb;text-align:center'>{act_labels[idx]}</div>", unsafe_allow_html=True)

        st.markdown("---")

        # ── Voice traits ──────────────────────────────────────────────────────
        st.markdown("<div class='section-label'>Voice traits</div>", unsafe_allow_html=True)

        trait_cols = st.columns(2)
        for i, (key, val) in enumerate(traits.items()):
            with trait_cols[i % 2]:
                label = TRAIT_LABELS[key]
                st.markdown(f"**{label}**")
                st.progress(val)
                st.caption(trait_desc(key, val))
                st.markdown("")

        st.markdown("---")

        # ── Radar chart ───────────────────────────────────────────────────────
        st.markdown("<div class='section-label'>Style fingerprint — this film vs director averages</div>", unsafe_allow_html=True)

        trait_keys  = ['Economy','Intensity','Interrogation','Emotion']
        trait_names = [TRAIT_LABELS[k] for k in trait_keys]
        film_vals   = [traits[k] for k in trait_keys]

        fig_radar = go.Figure()

        # Runner-up first (background)
        if runner in DIRECTOR_AVGS:
            rv = [DIRECTOR_AVGS[runner][k] for k in trait_keys]
            fig_radar.add_trace(go.Scatterpolar(
                r=rv + [rv[0]], theta=trait_names + [trait_names[0]],
                fill='toself', name=f'{runner}',
                line_color=COLORS.get(runner,'#aaa'),
                fillcolor='rgba(0,0,0,0)',
                line_dash='dot', line_width=1,
                opacity=0.6,
            ))

        # Winner average
        if winner in DIRECTOR_AVGS:
            wv = [DIRECTOR_AVGS[winner][k] for k in trait_keys]
            fig_radar.add_trace(go.Scatterpolar(
                r=wv + [wv[0]], theta=trait_names + [trait_names[0]],
                fill='toself', name=f'{winner} avg',
                line_color=color,
                fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)',
                line_dash='dash', line_width=1.5,
            ))

        # This film
        fig_radar.add_trace(go.Scatterpolar(
            r=film_vals + [film_vals[0]], theta=trait_names + [trait_names[0]],
            fill='toself', name='This film',
            line_color=color,
            fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)',
            line_width=2.5,
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0,1]),
                angularaxis=dict(tickfont=dict(size=12)),
            ),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
            height=360,
            margin=dict(l=60, r=60, t=20, b=60),
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

        st.markdown("---")

        # ── The reading ───────────────────────────────────────────────────────
        st.markdown("<div class='section-label'>The reading</div>", unsafe_allow_html=True)

        # Generate contextual explanation
        econ_word  = 'sparse' if traits['Economy'] > 0.6 else ('verbose' if traits['Economy'] < 0.35 else 'balanced')
        intens_word= 'rapid-fire' if traits['Intensity'] > 0.6 else ('unhurried' if traits['Intensity'] < 0.35 else 'fluid')
        conf_word  = 'strongly' if wp > 0.6 else ('tentatively' if wp < 0.35 else 'confidently')

        explanation = (
            f"The model {conf_word} identifies this film with **{winner}** ({round(wp*100)}% confidence). "
            f"The dialogue is {econ_word} — "
            f"{'a narrow vocabulary used repeatedly' if traits['Economy'] < 0.35 else 'words chosen with care' if traits['Economy'] > 0.6 else 'neither spare nor excessive'}. "
            f"Exchanges are {intens_word}. "
            f"The closest alternative is **{runner}** ({round(rp*100)}%) — "
            f"{'the gap is wide, this is a confident call' if wp - rp > 0.3 else 'the gap is narrow, suggesting these directors share stylistic territory'}."
        )
        st.markdown(explanation)

# ── Sidebar: Add a director ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Add a director")
    st.caption("Expand the model with your own director. Type their name and 3–5 films.")

    new_dir = st.text_input("Director name", placeholder="e.g. Wong Kar-wai")
    new_films_raw = st.text_area("Films (one per line)", placeholder="In the Mood for Love\nChungking Express\nHappy Together")

    if st.button("Add to model") and new_dir and new_films_raw:
        film_titles = [f.strip() for f in new_films_raw.strip().split('\n') if f.strip()]
        new_rows    = []
        prog        = st.progress(0)
        for i, title in enumerate(film_titles):
            feats, src, err = lookup_film(title)
            if feats and not err:
                new_rows.append({**{k: feats[k] for k in FEATURES}, 'label': new_dir})
                st.write(f"✓ {title}")
            else:
                st.write(f"✗ {title} — {err}")
            prog.progress((i+1)/len(film_titles))

        if new_rows:
            from sklearn.ensemble import RandomForestClassifier
            base   = model_data['dataset']
            all_df = pd.DataFrame(base + new_rows)
            X      = scaler.transform(all_df[FEATURES].values)
            y      = all_df['label'].values
            new_rf = RandomForestClassifier(n_estimators=200, random_state=42)
            new_rf.fit(X, y)
            st.session_state['custom_model'] = new_rf
            st.success(f"Model updated — {new_dir} added with {len(new_rows)} films")
            st.caption("This persists for your session only.")

    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "Trained on 25 films across 5 directors. "
        "60% leave-one-out accuracy vs 20% baseline. "
        "Features: vocabulary richness, pacing, burst patterns, silence structure. "
        "Built with scikit-learn + Streamlit."
    )
