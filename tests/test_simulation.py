from dataclasses import replace
from pathlib import Path
import json
import h5py
import numpy as np
import pytest
from src.config import Config,preset_config,resolve
from src.metrics import center,encounter_metrics
from src.simulation import simulate,save_run,load_summary
from galaxy import build_initial_conditions,_sample_nfw_radius
from octree import compute_accelerations,compute_accelerations_direct
from simulate import parse_args

def test_cli_preset_and_explicit_overrides(tmp_path):
    args=parse_args(["--preset","head_on","--andromeda-transverse-kms","70","--mw-mass","2e12",
                     "--N","20","--dt","1e6","--duration-gyr",".01","--softening","8"])
    c=resolve(args)
    assert c.andromeda_transverse_kms==70 and c.mw_mass==2e12
    assert c.N==20 and c.steps==10 and c.softening==8
    assert preset_config("head_on").andromeda_transverse_kms==0
    assert preset_config("high_mass_andromeda").m31_mass>Config().m31_mass
    path=tmp_path/"config.json";path.write_text(json.dumps({"N":30,"seed":9}))
    assert resolve(parse_args(["--config",str(path),"--N","40"])).N==40

def test_seed_masses_and_requested_relative_state():
    c=replace(Config(),N=40,mw_mass=1.2e12,m31_mass=1.9e12)
    p,v,m=build_initial_conditions(c);p2,v2,m2=build_initial_conditions(c)
    np.testing.assert_array_equal(p,p2);np.testing.assert_array_equal(v,v2)
    assert m[:c.N].sum()==pytest.approx(c.mw_mass)
    assert m[c.N:].sum()==pytest.approx(c.m31_mass)
    np.testing.assert_allclose(center(p[c.N:],m[c.N:])-center(p[:c.N],m[:c.N]),
                               [c.initial_distance_kpc,0,0],atol=1e-12)
    from src.config import KMS_TO_KPCYR
    np.testing.assert_allclose((center(v[c.N:],m[c.N:])-center(v[:c.N],m[:c.N]))/KMS_TO_KPCYR,
                               [c.andromeda_radial_kms,c.andromeda_transverse_kms,0],atol=1e-12)
    assert not np.array_equal(p,build_initial_conditions(replace(c,seed=c.seed+1))[0])

def test_nfw_enclosed_mass_distribution():
    r=_sample_nfw_radius(100000,20,200,np.random.default_rng(1))
    expected=(np.log(2)-.5)/(np.log(11)-10/11)
    assert np.mean(r<20)==pytest.approx(expected,abs=.003)

def test_force_symmetry_and_tree_accuracy():
    rng=np.random.default_rng(4);p=rng.normal(size=(30,3));m=rng.uniform(1e10,2e10,30)
    direct=compute_accelerations_direct(p,m,1)
    np.testing.assert_allclose((direct*m[:,None]).sum(axis=0),0,atol=1e-17)
    np.testing.assert_allclose(compute_accelerations(p,m,1,.01),direct,rtol=1e-12,atol=1e-26)

def test_diagnostics_final_snapshot_and_persistence(tmp_path):
    c=replace(Config(),N=20,steps=11,sample_every=4)
    d,m=simulate(c)
    assert d["time"][-1]==11*c.dt
    assert len(d["time"])==4
    assert np.isfinite(d["energy_drift"]).all()
    assert m["max_momentum_drift_relative"]<1e-12
    assert m["max_com_drift_kpc"]<1e-8
    np.testing.assert_allclose(d["separation"],np.linalg.norm(d["com_mw"]-d["com_m31"],axis=1))
    path=tmp_path/"run.h5";save_run(path,c,d,m)
    summary,metrics,config=load_summary(path)
    assert "pos" not in summary and metrics==m and config["seed"]==c.seed
    with h5py.File(path) as hf:
        np.testing.assert_array_equal(hf["pos"][:],d["pos"])
        assert json.loads(hf.attrs["provenance_json"])["source_sha256"]
    assert path.with_suffix(".config.json").exists()

def test_closest_approach_and_initial_threshold():
    m=encounter_metrics(np.array([0,1e9,2e9]),np.array([50,10,70]),40)
    assert m["min_com_separation_kpc"]==10 and m["min_com_separation_time_gyr"]==1
    assert m["first_close_approach_time_gyr"]==1 and not m["closest_at_boundary"]
    assert encounter_metrics(np.array([0,1]),np.array([10,20]),40)["first_close_approach_time_gyr"]==0
    assert encounter_metrics(np.array([0,1]),np.array([50,60]),40)["first_close_approach_time_gyr"] is None

def test_validation_disabled_not_reported_as_zero():
    d,m=simulate(replace(Config(),N=10,steps=2,validate=False),retain_particles=False)
    assert "pos" not in d and "energy" not in d and m["max_energy_drift_pct"] is None

@pytest.mark.parametrize("change",[{"N":2},{"dt":0},{"softening":-1},{"steps":0},
                                    {"dispersion":-1},{"mw_mass":float("nan")},{"seed":-1}])
def test_invalid_config(change):
    with pytest.raises(ValueError):replace(Config(),**change)

def test_ensemble_and_generated_figures(tmp_path):
    from src.sweeps import ensemble
    from src.visualization import make_figures,make_animation
    c=replace(Config(),N=10,steps=4,sample_every=2)
    rows,summary=ensemble(c,2,tmp_path/"mc",distributions={"andromeda_transverse_kms":{"mean":17,"std":2}})
    assert len(rows)==2 and summary[0]["min_com_separation_kpc_p05"]<=summary[0]["min_com_separation_kpc_p95"]
    assert (tmp_path/"mc"/"uncertainty.png").exists()
    d,m=simulate(c);make_figures(d,m,tmp_path/"figures");make_animation(d,tmp_path/"animation.gif",2)
    assert len(list((tmp_path/"figures").glob("*.png")))==4
    assert (tmp_path/"animation.gif").stat().st_size>0
    with pytest.raises(ValueError):
        ensemble(c,1,tmp_path/"bad",distributions={"initial_distance_kpc":{"mean":-1,"std":0}})

def test_streamlit_load_preset_submit_and_compare(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import src.simulation as module
    original=module.save_run
    saved=[]
    def capture(path,config,data,metrics):
        # Redirect test output and record the submitted config.
        target=tmp_path/f"test{len(saved)}.h5"
        original(target,config,data,metrics)
        saved.append(config)
    monkeypatch.setattr(module,"save_run",capture)
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"streamlit_app.py"),default_timeout=30).run()
    assert not app.exception
    assert not saved
    app.selectbox(key="preset").select("head_on")
    app.button(key="load_preset").click().run()
    assert app.number_input(key="cfg_andromeda_transverse_kms").value==0
    app.number_input(key="cfg_N").set_value(10)
    app.number_input(key="cfg_steps").set_value(4)
    app.number_input(key="cfg_sample_every").set_value(2)
    app.run()
    assert not saved
    app.number_input(key="cfg_N").set_value(10)
    app.number_input(key="cfg_steps").set_value(4)
    app.number_input(key="cfg_sample_every").set_value(2)
    app.button(key="run_simulation").click().run()
    assert not app.exception and len(saved)==1 and saved[0].N==10

    if len(app.multiselect[0].options)>=2:
        app.multiselect[0].select(app.multiselect[0].options[1]).run()
        assert not app.exception
