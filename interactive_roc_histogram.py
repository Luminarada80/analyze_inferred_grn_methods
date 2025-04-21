import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from bokeh.layouts import row, column
from bokeh.models import (
    ColumnDataSource, Slider, Span, Div, CustomJS, Select, Whisker, HoverTool, Title
)
from bokeh.plotting import figure, curdoc
import panel as pn

# Initialize Panel with Bokeh extension
pn.extension('bokeh')

# ─── 1) Define your dataset selector ────────────────────────────────────────
dataset_options = [
    "mESC_RN112_logof",
    "mESC_RN111_chipseq",
    "macrophage",
]
dataset_folder = dataset_options[0]  # initial value
dataset_select = Select(
    title="Dataset:",
    value=dataset_folder,
    options=dataset_options,
)

spinner = pn.indicators.LoadingSpinner(
    name='Loading', value=False, visible=False,
    width=50, height=50,
    # you can still give it a class if you like, but it's not needed:
    # css_classes=['spinner-overlay']
)
# apply inline styles so it always floats in the exact center:
spinner.styles = {
    'position' : 'fixed',
    'top'      : '45%',
    'left'     : '40%',
    'transform': 'translate(-40%, -45%)',
    'z-index'  : '10000',
}

# ─── 2) Factor out your data‑loading + precompute logic ────────────────────
def load_and_precompute(folder):
    gt = pd.read_csv(f"./{folder}/balanced_ground_truth.csv")
    inf = pd.read_csv(f"./{folder}/balanced_inferred_network.csv")
    df = pd.concat([gt, inf], ignore_index=True)
    y_true = df["true_interaction"].values
    y_scores = df["Score"].values

    # ROC & PR
    fpr, tpr, _   = roc_curve(y_true, y_scores)
    prec, rec, _  = precision_recall_curve(y_true, y_scores)
    
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    
    # randomized uniform scores
    rand_scores        = np.random.uniform(0, 1, size=len(y_scores))
    r_fpr, r_tpr, _    = roc_curve(y_true, rand_scores)
    r_prec, r_rec, _   = precision_recall_curve(y_true, rand_scores)
    rand_auroc         = roc_auc_score(y_true, rand_scores)
    rand_auprc         = average_precision_score(y_true, rand_scores)
    
    # histogram bins
    bins       = np.linspace(0, 1, 50)
    centers    = (bins[:-1] + bins[1:]) / 2
    idx        = np.clip(np.digitize(y_scores, bins) - 1, 0, len(centers)-1)
    tp_counts  = np.bincount(idx[y_true==1], minlength=len(centers))
    fp_counts  = np.bincount(idx[y_true==0], minlength=len(centers))
    
    thresh = 0.5
    raw = {
        "TP": y_scores[(y_true==1) & (y_scores>=thresh)],
        "FP": y_scores[(y_true==0) & (y_scores>=thresh)],
        "TN": y_scores[(y_true==0) & (y_scores< thresh)],
        "FN": y_scores[(y_true==1) & (y_scores< thresh)],
    }
    # compute boxplot stats
    box = {"class":[],"q1":[],"q2":[],"q3":[],"lower":[],"upper":[]}
    for cls, arr in raw.items():
        cls = str(cls)
        q1,q2,q3 = np.percentile(arr, [25,50,75])
        iqr      = q3 - q1
        low_wh   = max(arr.min(), q1-1.5*iqr)
        high_wh  = min(arr.max(), q3+1.5*iqr)
        box["class"].append(cls)
        box["q1"].append(q1)
        box["q2"].append(q2)
        box["q3"].append(q3)
        box["lower"].append(low_wh)
        box["upper"].append(high_wh)

    return {
        "bins":       bins,
        "centers":    centers,
        "tp_counts":  tp_counts,
        "fp_counts":  fp_counts,
        "total_pos":  int((y_true==1).sum()),
        "total_neg":  int((y_true==0).sum()),
        "roc":        dict(fpr=fpr, tpr=tpr),
        "pr":         dict(recall=rec, precision=prec),
        "rand_roc":   dict(fpr=r_fpr, tpr=r_tpr),
        "rand_pr":    dict(recall=r_rec, precision=r_prec),
        "box":        box,
        "auroc":      auroc,
        "auprc":      auprc,
        "rand_auroc": rand_auroc,
        "rand_auprc": rand_auprc,
    }

# first load
data = load_and_precompute(dataset_folder)

# create the box_source
box_source = ColumnDataSource(data=data["box"])

# 3) build the box_plot figure and keep the vbar renderer
box_fig = figure(
    x_range=["TP","FP","TN","FN"],
    width=400, height=300,
    title="Score Boxplot",
    tools=""
)

# draw the boxes and grab the renderer
box_renderer = box_fig.vbar(
    x="class", width=0.7,
    top="q3", bottom="q1",
    source=box_source,
    fill_alpha=0.3,
    line_color="black",
)

# median line
box_fig.segment(
    x0="class", y0="q2",
    x1="class", y1="q2",
    source=box_source,
    line_width=2,
    line_color="black"
)

# whiskers
whisker = Whisker(source=box_source, base="class",
                  upper="upper", lower="lower")
box_fig.add_layout(whisker)

# 4) now create and add the hover tool for the boxes
box_hover = HoverTool(
    renderers=[box_renderer],
    tooltips=[
        ("Class",       "@class"),
        ("Upper whisker","@upper{0.000}"),
        ("Q3",          "@q3{0.000}"),
        ("Median (Q2)", "@q2{0.000}"),
        ("Q1",          "@q1{0.000}"),
        ("Lower whisker", "@lower{0.000}")
    ]
)
box_fig.add_tools(box_hover)

# ─── 3) Create your ColumnDataSources once ────────────────────────────────
hist_source = ColumnDataSource(data=dict(
    x   = data["centers"].tolist(),
    tp  = data["tp_counts"].tolist(),
    fp  = data["fp_counts"].tolist(),
    tn  = [0]*len(data["centers"]),
    fn  = [0]*len(data["centers"]),
))
roc_source = ColumnDataSource(data=data["roc"])
pr_source  = ColumnDataSource(data=data["pr"])

# sources for the random curves
rand_roc_source = ColumnDataSource(data=data["rand_roc"])
rand_pr_source  = ColumnDataSource(data=data["rand_pr"])

roc_fig = figure(width=500, height=400, title="ROC Curve",
                 x_axis_label="False Positive Rate", y_axis_label="True Positive Rate", tools="")
roc_fig.line('fpr', 'tpr', source=roc_source, line_width=2, color='navy')
roc_fig.line([0,1], [0,1], color='gray', line_dash='dashed')
roc_threshold_line = Span(location=0.5, dimension='height',
                          line_color='black', line_dash='dashed', line_width=2)
roc_fig.add_layout(roc_threshold_line)

pr_fig = figure(width=500, height=400,
                title="Precision‑Recall Curve",
                x_axis_label="Recall", y_axis_label="Precision", tools="")
pr_fig.line('recall', 'precision', source=pr_source, line_width=2, color='green')
pr_threshold_line = Span(location=0.5, dimension='height',
                         line_color='black', line_dash='dashed', line_width=2)
pr_fig.add_layout(pr_threshold_line)

# existing “real” metric titles
roc_metric = Title(text=f"AUROC: {data['auroc']:.3f}", align="center")
pr_metric  = Title(text=f"AUPRC: {data['auprc']:.3f}", align="center")
roc_fig.add_layout(roc_metric, 'below')
pr_fig.add_layout(pr_metric, 'below')

# NEW: random metric titles, added *below* the real ones
rand_roc_metric = Title(text=f"Random: {data['rand_auroc']:.3f}", align="center")
rand_pr_metric  = Title(text=f"Random: {data['rand_auprc']:.3f}", align="center")
roc_fig.add_layout(rand_roc_metric, 'below')
pr_fig.add_layout(rand_pr_metric, 'below')

# ─── add the dashed glyphs ────────────────────────────────────────────────
roc_fig.line(
    'fpr', 'tpr', source=rand_roc_source,
    line_dash='dashed', color='gray', line_width=2,
)
pr_fig.line(
    'recall', 'precision', source=rand_pr_source,
    line_dash='dashed', color='gray', line_width=2,
)

hist_fig = figure(width=600, height=300, title="Score Distribution",
                   tools="")
# hist_fig.vbar(x='x', top='tn', width=0.015, color="#b6cde0", fill_alpha=0.7, source=hist_source)
# hist_fig.vbar(x='x', top='fn', width=0.015, color="#efc69f", fill_alpha=0.7, source=hist_source)
hist_fig.vbar(x='x', top='fp', width=0.015, color="#4195df", fill_alpha=0.7, source=hist_source, legend_label="Negative Score")
hist_fig.vbar(x='x', top='tp', width=0.015, color="#dc8634", fill_alpha=0.7, source=hist_source, legend_label="Positive Score")

threshold_line = Span(location=0.5, dimension='height',
                      line_color='black', line_dash='dashed', line_width=2)
hist_fig.add_layout(threshold_line)
# pull out the auto–created legend
legend = hist_fig.legend[0]

# put it BELOW the plot, outside the frame:
hist_fig.add_layout(legend, 'below')

# make it horizontal …
legend.orientation       = 'horizontal'
# … and center its labels under each glyph
legend.label_text_align  = 'center'
legend.margin = 0
legend.padding = 5  # or whatever spacing you like

# 2) Create your throttled slider in Python
slider = Slider(
    start=0.0, end=1.0, value=0.5, step=0.01,
    title="Score Threshold"
)

# 3.5) Create a SECOND source to hold the un‐thresholded bin data
raw_source = ColumnDataSource(data=dict(
    centers = data["centers"].tolist(),
    raw_tp  = data["tp_counts"].tolist(),
    raw_fp  = data["fp_counts"].tolist(),
))

# 4) Redefine your JS callback to take raw_source instead of raw Python arrays
slider.js_on_change('value', CustomJS(args=dict(
    threshold_line=threshold_line,
    roc_line=roc_threshold_line,
    pr_line=pr_threshold_line,
    raw_source=raw_source
), code="""
    const θ = cb_obj.value;
    const centers   = raw_source.data['centers'];
    const tp_counts = raw_source.data['raw_tp'];
    const fp_counts = raw_source.data['raw_fp'];
    const n         = centers.length;
    
    // 1) move the histogram thresh‐line
    threshold_line.location = θ;

    // total positives/negatives (constant)
    const total_pos = tp_counts.reduce((a,b)=>a+b,0);
    const total_neg = fp_counts.reduce((a,b)=>a+b,0);

    // find first bin ≥ θ
    let k = centers.findIndex(c => c >= θ);

    // 1) if θ is below your first center, everything is “above” threshold:
    //    FPR = FP_above / total_neg = 1,  Recall = TP_above / total_pos = 1
    if (k === 0) {
    roc_line.location = 1;
    pr_line.location  = 1;

    // 2) if no bin is ≥ θ (θ > max center), nothing is “above” threshold:
    //    FPR = 0, Recall = 0
    } else if (k === -1) {
    roc_line.location = 0;
    pr_line.location  = 0;

    // 3) the normal case
    } else {
    const fp_above = fp_counts.slice(k).reduce((a,b)=>a+b,0);
    const tp_above = tp_counts.slice(k).reduce((a,b)=>a+b,0);
    roc_line.location = fp_above / total_neg;
    pr_line.location  = tp_above / total_pos;
    }
"""))


# 5) Hook up the slider to the throttled event
# slider.js_on_change('value', callback)

# 6) In your Python update_dataset callback, *also* update raw_source
def update_dataset(attr, old, new):
    spinner.value = True
    spinner.visible = True
    
    def do_update():
        d = load_and_precompute(new)

        # histogram display source
        hist_source.data.update({
        "x":  d["centers"].tolist(),
        "tp": d["tp_counts"].tolist(),
        "fp": d["fp_counts"].tolist(),
        "tn": [0]*len(d["centers"]),
        "fn": [0]*len(d["centers"]),
        })
        # ROC & PR
        roc_source.data.update(d["roc"])
        pr_source.data.update(d["pr"])
        
        # Update the random‐curve sources
        rand_roc_source.data.update(d["rand_roc"])
        rand_pr_source.data.update(d["rand_pr"])

        # **also** swap in the raw arrays for future JS thresholding
        raw_source.data.update({
        "centers": d["centers"].tolist(),
        "raw_tp":  d["tp_counts"].tolist(),
        "raw_fp":  d["fp_counts"].tolist(),
        })
        
        roc_metric.text      = f"AUROC: {d['auroc']:.3f}"
        rand_roc_metric.text = f"random AUROC: {d['rand_auroc']:.3f}"

        pr_metric.text       = f"AUPRC: {d['auprc']:.3f}"
        rand_pr_metric.text  = f"random AUPRC: {d['rand_auprc']:.3f}"

        box_source.data = load_and_precompute(new)["box"]
        
        # reset spans if desired
        slider.value                = 0.5
        threshold_line.location     = 0.5
        roc_threshold_line.location = 0.5
        pr_threshold_line.location  = 0.5
        
        spinner.value = False
        spinner.visible = False
    
    curdoc().add_next_tick_callback(do_update)

dataset_select.on_change('value', update_dataset)
update_dataset(None, None, dataset_options[0])


# create a “separator” Div that renders as a tall, thin vertical line
separator = Div(
    text="", 
    width=2, 
    styles={
        "border-left": "2px solid #888", 
        "height": "25px",     # adjust to match your two-figure stack height
        "margin": "10px 0"
    }
)


# ─── 6) Panel layout ─────────────────────────────────────────────────────
dashboard = pn.Column(
    pn.Row(dataset_select),            # spinner removed here
    pn.Row(roc_fig, pr_fig),      # ← panes here, not roc_fig/pr_fig
    pn.Row(hist_fig, box_fig),
    slider,
)

# stack dashboard and overlay‑spinner
pn.Column(dashboard, spinner).servable(
    title="AUROC + Histogram Dashboard"
)