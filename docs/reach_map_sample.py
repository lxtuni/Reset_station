# 采样 SO-101 在桅杆抓取高度（夹爪尖端 z 0..0.04 m）能到的位置；结果存 docs/reach_samples.npz，再用 reach_map_plot.py 画图
# 运行：在项目根目录 python docs/reach_map_sample.py（约 1 分钟）
import numpy as np, mujoco
m=mujoco.MjModel.from_xml_path("assets/robotstudio_so101/reset_station.xml"); d=mujoco.MjData(m)
na=6; s=m.site("gripperframe").id; j=m.joint("wrist_roll").id; bj=m.jnt_bodyid[j]
jid=m.joint("body2_free").id; qa=m.jnt_qposadr[jid]
walls={m.geom(n).id for n in ["tank_wall_back","tank_wall_front","tank_wall_yneg","tank_wall_ypos","tank_bottom","table"]}
b2={m.geom("body2_hull").id,m.geom("body2_mast").id}
arm={g for g in range(m.ngeom) if m.geom_bodyid[g]!=0 and g not in b2}
lo,hi=m.jnt_range[:na,0],m.jnt_range[:na,1]; rng=np.random.default_rng(0)
A=[];V=[]
for i in range(400000):
    d.qpos[:na]=rng.uniform(lo,hi); d.qpos[qa:qa+7]=[5,5,0,1,0,0,0]; mujoco.mj_forward(m,d)
    p=d.site_xpos[s]
    if not (0.0<=p[2]<=0.04): continue
    bad=False
    for c in d.contact[:d.ncon]:
        g1,g2=c.geom1,c.geom2
        if (g1 in arm or g2 in arm) and not (g1 in walls or g2 in walls): bad=True;break
    if bad: continue
    ax=-(d.xmat[bj].reshape(3,3)@m.jnt_axis[j])
    A.append(p[:2].copy())
    if ax[2]<-0.94: V.append(p[:2].copy())
A=np.array(A);V=np.array(V); np.savez("docs/reach_samples.npz",A=A,V=V)
r=np.hypot(V[:,0],V[:,1]); print("any-orient pts",len(A),"vertical pts",len(V)," vertical r range",r.min().round(3),r.max().round(3), " x>0.085 r max", r[V[:,0]>0.085].max().round(3))
