#!/usr/bin/env python3
"""Generate arXiv-style paper figures -> paper/figures/*.png.
Numbers are the verified full-context (ARTICLE_CHARS=100000) study results.
Run: python3 paper/figs.py"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})
TEACHER="#b45309"; TUNED="#15803d"; FEWSHOT="#1d4ed8"; CONSTR="#64748b"; BASE="#b0b7c3"
BLUE="#2563eb"; GREEN="#15803d"; RED="#b91c1c"

# ---- Fig 1: content-analysis panel mockup ----
def fig1():
    fig, ax = plt.subplots(figsize=(6.6, 3.5)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,6)
    ax.add_patch(FancyBboxPatch((0.2,0.2),9.6,5.6, boxstyle="round,pad=0.05,rounding_size=0.15",
                 fc="#f8fafc", ec="#cbd5e1", lw=1.2))
    ax.text(0.55,5.25,"Content Analysis", fontsize=13, fontweight="bold", color="#0f172a")
    ax.text(0.55,4.55,'"Industry cooperation on AI-safety norms is essential to counter\ncompetitive pressures that erode caution; the paper proposes four\nstrategies and argues long-term safety outcomes beat short-term gains."',
            fontsize=8.2, color="#334155", style="italic", va="top")
    badges=[("sentiment","positive","#16a34a"),("urgency","developing","#d97706"),
            ("frame","analytical","#2563eb"),("tone","analytical","#7c3aed"),("depth","standard","#0891b2")]
    x=0.55
    for name,val,c in badges:
        w=0.28+0.083*len(name+val)
        ax.add_patch(FancyBboxPatch((x,2.35),w,0.5, boxstyle="round,pad=0.02,rounding_size=0.1", fc=c, ec="none", alpha=0.16))
        ax.text(x+0.12,2.6,f"{name}: ",fontsize=7.5,color="#475569",va="center")
        ax.text(x+0.12+0.052*len(name+": "),2.6,val,fontsize=7.5,fontweight="bold",color=c,va="center")
        x+=w+0.2
    ax.text(0.55,1.65,"Topics", fontsize=8.5, fontweight="bold", color="#0f172a")
    x=0.55
    for t in ["AI Safety","Industry Cooperation","Responsible AI"]:
        w=0.3+0.09*len(t)
        ax.add_patch(FancyBboxPatch((x,0.85),w,0.5, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#e2e8f0", ec="#94a3b8", lw=0.6))
        ax.text(x+w/2,1.1,t,fontsize=7.5,color="#334155",ha="center",va="center"); x+=w+0.25
    fig.savefig(os.path.join(OUT,"fig1_panel.png")); plt.close(fig)

# ---- Fig 2: experiment flow ----
def fig2():
    fig, ax = plt.subplots(figsize=(10,2.9)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,3)
    steps=[("500 articles\nteacher-labeled",BASE),("Stratified split\n401 train / 93 test","#cbd5e1"),
           ("QLoRA fine-tune\nQwen3-0.6B ×3 seeds",TUNED),("7 arms @ T=0\nteacher·base·2 controls·3 tuned","#cbd5e1"),
           ("3 sub-tasks\nsummary·labels·topics","#cbd5e1"),("2-judge panel\n+ neg. control",FEWSHOT),
           ("Bootstrap stats\ngap-closure",TEACHER)]
    n=len(steps); w=1.24; gap=(10-n*w)/(n+1)
    for i,(txt,c) in enumerate(steps):
        x=gap+i*(w+gap)
        ax.add_patch(FancyBboxPatch((x,1.1),w,0.9, boxstyle="round,pad=0.03,rounding_size=0.08",
                     fc=c, ec="#334155", lw=0.9, alpha=0.85 if c in(TUNED,FEWSHOT,TEACHER) else 1))
        ax.text(x+w/2,1.55,txt,fontsize=6.6,ha="center",va="center",
                color="white" if c in(TUNED,FEWSHOT,TEACHER) else "#0f172a", fontweight="bold")
        if i<n-1:
            ax.add_patch(FancyArrowPatch((x+w,1.55),(x+w+gap,1.55),arrowstyle="-|>",mutation_scale=11,color="#475569",lw=1.1))
    ax.text(5,0.5,"temperature 0 · reference-free judging against the article · human-free on the critical path",
            fontsize=7.5,ha="center",color="#64748b",style="italic")
    fig.savefig(os.path.join(OUT,"fig2_flow.png")); plt.close(fig)

# ---- Fig 3: gap-closure toward the teacher (two panels) ----
def _gc(base,tuned,teacher): return (tuned-base)/(teacher-base)*100 if teacher!=base else float("nan")
def fig3():
    summ=[("faithful",79.6,75.6,93.5),("length",61.3,59.1,81.7),("tech-lens",36.6,54.5,63.4),
          ("teacher-lens",22.6,62.0,87.1),("takeaway",44.1,74.2,89.2),("tone",59.1,82.8,92.5),
          ("thesis",84.9,94.2,97.8),("topics",87.1,95.7,98.9),("opening",51.6,77.8,73.1),("SUMMARY macro",55.0,72.5,84.8)]
    clas=[("tone-label",24.4,29.0,78.5),("depth",43.0,48.8,53.8),("frame",50.0,58.1,66.7),
          ("sentiment",43.0,66.0,80.6),("urgency",62.8,78.5,57.0),("CLASS. macro",44.7,56.0,67.3)]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(11,5.2),gridspec_kw={"width_ratios":[1,0.72]})
    def draw(ax,data,title):
        labs=[d[0] for d in data]; y=np.arange(len(data))
        vals=[]; cols=[]
        for _,b,t,te in data:
            if t>te: g=_gc(b,t,te) if te>b else 105; cols.append(GREEN)      # beats teacher
            elif t<b: g=_gc(b,t,te); cols.append(RED)                        # below base
            else: g=_gc(b,t,te); cols.append(BLUE)
            vals.append(max(min(g,120),-40) if g==g else 105)
        ax.barh(y,vals,color=cols,zorder=3,height=0.62)
        ax.axvline(0,color="#334155",lw=0.8); ax.axvline(100,color="#334155",lw=0.8,ls="--",alpha=0.6)
        for yi,(v,d) in enumerate(zip(vals,data)):
            b,t,te=d[1],d[2],d[3]
            lab = "beats teacher" if t>te and te<=b else f"{_gc(b,t,te):.0f}%"
            ax.text(v+(3 if v>=0 else -3),yi,lab,va="center",ha="left" if v>=0 else "right",fontsize=7.5,
                    color=cols[yi],fontweight="bold")
        ax.set_yticks(y); ax.set_yticklabels(labs,fontsize=8.5); ax.set_xlim(-45,132)
        ax.set_xlabel("gap-closure: base=0%, teacher=100%",fontsize=8.5); ax.set_title(title,fontsize=10.5,fontweight="bold")
        ax.grid(axis="y",alpha=0)
    draw(a1,summ,"Summarization (8 checks + topics)")
    draw(a2,clas,"Classification (5 fields)")
    fig.suptitle("Gap-closure toward the teacher, per axis  (green = beat teacher · red = below base · full-text grading)",
                 fontsize=11,y=1.02)
    fig.savefig(os.path.join(OUT,"fig3_gapclosure.png")); plt.close(fig)

# ---- Fig 4: checklist pass-rate per arm + CI ----
def fig4():
    arms=["Base\nzero-shot","Base +\nconstrained","Base +\nfew-shot","Tuned\n(distilled)","Teacher\nR1-8B"]
    vals=[55.0,57.0,66.7,72.5,84.8]; cis=[(49.5,60.6),(51.9,61.8),(62.0,71.2),(68.5,76.3),(81.0,88.0)]
    cols=[BASE,CONSTR,FEWSHOT,TUNED,TEACHER]
    err=[[v-lo for v,(lo,hi) in zip(vals,cis)],[hi-v for v,(lo,hi) in zip(vals,cis)]]
    fig,ax=plt.subplots(figsize=(7,4.1)); x=np.arange(len(arms))
    ax.bar(x,vals,color=cols,width=0.62,zorder=3)
    ax.errorbar(x,vals,yerr=err,fmt="none",ecolor="#111",elinewidth=1.3,capsize=5,zorder=4)
    for xi,v in zip(x,vals): ax.text(xi,v+2.4,f"{v:.1f}",ha="center",fontsize=9.5,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(arms,fontsize=9); ax.set_ylabel("Summary checklist pass-rate (%)"); ax.set_ylim(0,100)
    ax.set_title("Summary quality by arm — full-text grading, N=93 (mean ± 95% CI)",fontsize=10.5)
    ax.annotate("beats both non-\ndistillation controls",xy=(3,72.5),xytext=(1.6,91),fontsize=8.5,color=TUNED,
                arrowprops=dict(arrowstyle="->",color=TUNED,lw=1.1),ha="center")
    fig.savefig(os.path.join(OUT,"fig4_checklist.png")); plt.close(fig)

# ---- Fig 5: batch economics + quality x cost ----
def fig5():
    fig,(axL,axR)=plt.subplots(1,2,figsize=(11,4.2))
    n=np.array([0,100,200,300,400,500])
    axL.plot(n,n*39.2/3600,"-o",color=TEACHER,lw=2,label="Teacher R1-8B (39.2 s/article)")
    axL.plot(n,n*0.774/3600,"-o",color=TUNED,lw=2,label="Distilled 0.6B (0.77 s/article)")
    axL.set_xlabel("Articles enriched"); axL.set_ylabel("Wall-clock (hours)"); axL.set_title("Batch cost vs. volume")
    axL.legend(fontsize=8,loc="upper left")
    axL.annotate("5.4 h",xy=(500,5.44),xytext=(360,4.6),color=TEACHER,fontsize=10,fontweight="bold")
    axL.annotate("~7 min",xy=(500,0.107),xytext=(300,0.95),color=TUNED,fontsize=10,fontweight="bold",
                 arrowprops=dict(arrowstyle="->",color=TUNED,lw=1))
    pts=[("Teacher",39200,84.8,TEACHER),("Tuned",774,72.5,TUNED),("Few-shot",856,66.7,FEWSHOT),
         ("Constrained",824,57.0,CONSTR),("Base",845,55.0,BASE)]
    for name,lat,q,c in pts:
        axR.scatter(lat,q,s=150,color=c,zorder=3,edgecolor="white",linewidth=1.2)
        axR.annotate(name,(lat,q),xytext=(0,8 if name!="Constrained" else -14),textcoords="offset points",
                     ha="center",fontsize=8.5,color=c,fontweight="bold")
    axR.set_xscale("log"); axR.set_xlabel("Latency per article (ms, log scale)")
    axR.set_ylabel("Summary quality (checklist %)"); axR.set_ylim(45,92); axR.set_title("Quality × cost plane")
    axR.text(1050,49,"all 0.6B arms ~0.8 s\n(speed is free with size)",fontsize=7.5,color="#666",ha="center")
    fig.suptitle("The batch collapse and the quality × cost trade-off",fontsize=11.5)
    fig.savefig(os.path.join(OUT,"fig5_economics.png")); plt.close(fig)

if __name__=="__main__":
    # NOTE: the paper's figures (study-at-a-glance, Fig 3 gap-closure, Fig 4 checklist,
    # Fig 5 efficiency) are AUTHOR-PROVIDED artifacts committed in paper/figures/ and are
    # NOT regenerated here — do not overwrite them. This script only (re)builds the
    # optional Figure 1 content-panel mock-up (fig1_panel.png). The fig2/3/4/5 functions
    # above are retained only as a record of the earlier auto-generated drafts.
    fig1()
    print("wrote fig1_panel to", OUT, "(other figures are author-provided)")
