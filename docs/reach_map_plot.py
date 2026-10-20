# 画有效抓取范围图 docs/reach_map.png（需要 matplotlib；中文字体用 Windows 的微软雅黑）
# 运行：在项目根目录 python docs/reach_map_plot.py
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib import font_manager as fm; fm.fontManager.addfont("/mnt/c/Windows/Fonts/msyh.ttc"); plt.rcParams["font.family"]=[fm.FontProperties(fname="/mnt/c/Windows/Fonts/msyh.ttc").get_name()]; plt.rcParams["axes.unicode_minus"]=False
D=np.load("docs/reach_samples.npz"); A,V=D["A"],D["V"]
def envelope(P,nb=72):
    th=np.arctan2(P[:,1],P[:,0]); r=np.hypot(P[:,0],P[:,1]); e=np.linspace(-np.pi,np.pi,nb+1); out=[];inn=[]
    for a,b in zip(e[:-1],e[1:]):
        s=(th>=a)&(th<b)
        if s.sum()<3: continue
        c=(a+b)/2; out.append((c,np.percentile(r[s],99))); inn.append((c,np.percentile(r[s],1)))
    return np.array(out),np.array(inn)
def poly(out,inn):
    o=np.c_[out[:,1]*np.cos(out[:,0]),out[:,1]*np.sin(out[:,0])]; i=np.c_[inn[:,1]*np.cos(inn[:,0]),inn[:,1]*np.sin(inn[:,0])][::-1]
    return np.r_[o,i]
oa,ia=envelope(A); ov,iv=envelope(V)
fig,ax=plt.subplots(figsize=(10,6.2),dpi=130)
ax.add_patch(Rectangle((-0.06,-0.5),1.0,1.0,fill=False,ls="--",lw=1.2,ec="#888",label="1 m × 1 m 缸（候选）"))
ax.add_patch(Rectangle((-0.06,-0.25),1.0,0.5,fill=False,ls="-.",lw=1.2,ec="#555",label="1 m × 0.5 m 缸（候选）"))
ax.add_patch(Rectangle((-0.06,-0.25),0.47,0.5,fill=False,lw=1.5,ec="#1f77b4",label="现在的缸 0.47 × 0.5 m"))
ax.add_patch(Polygon(poly(oa,ia),closed=True,fc="#9ecae1",ec="none",alpha=0.6,label="夹爪尖端能到（任意朝向）"))
ax.add_patch(Polygon(poly(ov,iv),closed=True,fc="#08519c",ec="none",alpha=0.75,label="能竖直朝下夹桅杆（可调头）"))
ax.add_patch(Rectangle((-0.06,-0.07),0.13,0.14,fc="#8c6d31",ec="none",label="平台 + 底座"))
ax.plot([0.0825,0.0825],[-0.25,0.25],color="#d62728",lw=2.5,label="隔板 x = 0.08")
for R in (0.2,0.3,0.38):
    t=np.linspace(-np.pi/2*1.2,np.pi/2*1.2,200); ax.plot(R*np.cos(t),R*np.sin(t),color="k",lw=0.5,alpha=0.35); ax.text(R*np.cos(0.95),R*np.sin(0.95),f"r = {R} m",fontsize=8,alpha=0.6)
ax.plot(0,0,"k+",ms=12)
ax.set_xlim(-0.12,1.0); ax.set_ylim(-0.55,0.55); ax.set_aspect("equal"); ax.grid(alpha=0.25)
ax.set_xlabel("x (m)  ← 机械臂在 x = 0，朝 +x 伸出"); ax.set_ylabel("y (m)")
ax.set_title("SO-101 立着装：桅杆抓取高度（z 0…0.04 m）的有效抓取范围（俯视）")
ax.legend(loc="upper right",fontsize=8,framealpha=0.9)
plt.tight_layout(); plt.savefig("docs/reach_map.png")
# area numbers
from matplotlib.path import Path
pv=Path(poly(ov,iv)); g=np.mgrid[0.085:0.94:0.005,-0.5:0.5:0.005].reshape(2,-1).T
inv=pv.contains_points(g); a_v=inv.sum()*0.005**2
sel=(g[:,0]>0.085+0.058)
print("vertical-grasp area beyond fence (m2): %.3f ; max x %.2f ; |y| max %.2f"%(a_v, g[inv,0].max(), np.abs(g[inv,1]).max()))
for W in (0.5,1.0):
    box=(g[:,0]<=0.94)&(np.abs(g[:,1])<=W/2); print("1 m x %.1f m tank: boat-zone area %.2f m2, covered %.0f%%"%(W,box.sum()*0.005**2,100*(inv&box).sum()/box.sum()))
