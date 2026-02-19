"""
FICUTS Domain Registry — 150-domain expansion system.

Maps 150 target learning domains to UnifiedTensorNetwork mode head indices.
Each domain is a potential expansion region: when the system encounters data
matching a domain, it activates HDV subspace dimensions for that domain and
promotes the corresponding dummy mode head.

The 150 domains follow the five expansion principles:
  1. Preallocate: all 150 domains are registered but mostly dormant at start
  2. Reserve dummy heads: mode_heads[13..149] are reserved for these domains
  3. Dynamic basis: activation calls _register_domain_dims on IntegratedHDVSystem
  4. Curvature-triggered: DynamicExpander activates domains on ρ-spike
  5. Spectral hashing: new Koopman frequencies hash to unused HDV indices
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── 150-domain manifest ────────────────────────────────────────────────────────
# (head_index, snake_case_id, display_name, keyword_set)
# head_index corresponds to mode_heads[head_index] in UnifiedTensorNetwork.
# Domains 0-12 overlap with the 13 named scientific heads (ece, biology, ...);
# domains 13-149 occupy the 137 previously-dummy heads.

DOMAIN_MANIFEST: List[Tuple[int, str, str, Set[str]]] = [
    (0,  "nonlinear_pde",          "Nonlinear PDE solving",
     {"pde","partial","differential","equation","physics","informed","navier","stokes","heat","wave","poisson","laplace"}),
    (1,  "neural_operators",       "Neural operators for PDE families",
     {"neural","operator","fourier","deeponet","parametric","pde","function","space"}),
    (2,  "surrogate_cfd",          "Surrogate modeling for CFD/FEA",
     {"surrogate","cfd","fea","fluid","dynamics","finite","element","mesh","flow"}),
    (3,  "turbulence_modeling",    "Turbulence closure modeling",
     {"turbulence","closure","reynolds","stress","les","rans","eddy","viscosity"}),
    (4,  "inverse_identification", "Inverse parameter identification",
     {"inverse","parameter","identification","calibration","estimation","infer"}),
    (5,  "chaotic_prediction",     "Chaotic system prediction",
     {"chaotic","chaos","lorenz","lyapunov","attractor","bifurcation","nonlinear"}),
    (6,  "multiscale_simulation",  "Multi-scale physical simulation",
     {"multiscale","multi","scale","coarse","fine","homogenization","molecular","continuum"}),
    (7,  "quantum_systems",        "Quantum system approximation",
     {"quantum","schrodinger","wavefunction","hamiltonian","entanglement","qubit","variational"}),
    (8,  "materials_discovery",    "Materials discovery and lattice properties",
     {"materials","lattice","crystal","property","discovery","dft","density","functional"}),
    (9,  "semiconductor_modeling", "Semiconductor device modeling",
     {"semiconductor","mosfet","transistor","doping","band","gap","carrier","drift"}),
    (10, "em_field",               "Electromagnetic field approximation",
     {"electromagnetic","em","field","maxwell","antenna","radiation","wave","propagation"}),
    (11, "circuit_nonlinear",      "Circuit nonlinear component modeling",
     {"circuit","nonlinear","diode","spice","analog","amplifier","resistor","capacitor"}),
    (12, "rf_signal",              "RF and signal propagation modeling",
     {"rf","radio","signal","propagation","channel","fading","frequency","modulation"}),
    (13, "power_grid",             "Power grid stability prediction",
     {"power","grid","stability","voltage","frequency","fault","transmission","bus"}),
    (14, "smart_grid",             "Smart grid optimization",
     {"smart","grid","demand","response","distribution","microgrid","dispatch","optimal"}),
    (15, "renewable_energy",       "Renewable energy output forecasting",
     {"renewable","solar","wind","photovoltaic","forecast","irradiance","turbine","generation"}),
    (16, "battery_degradation",    "Battery degradation modeling",
     {"battery","degradation","capacity","fade","cycle","lithium","soc","soh","aging"}),
    (17, "thermal_optimization",   "Thermal system optimization",
     {"thermal","heat","temperature","cooling","conduction","convection","radiation","management"}),
    (18, "structural_health",      "Structural health monitoring",
     {"structural","health","monitoring","vibration","damage","detection","fatigue","crack"}),
    (19, "predictive_maintenance", "Predictive maintenance for industrial systems",
     {"predictive","maintenance","industrial","fault","failure","bearing","sensor"}),
    (20, "robotics_control",       "Autonomous robotics control",
     {"robot","robotic","control","trajectory","manipulation","arm","actuator","autonomous"}),
    (21, "multi_agent_robotics",   "Multi-agent robotic coordination",
     {"multi","agent","robot","coordination","swarm","formation","cooperative","distributed"}),
    (22, "aerospace_trajectory",   "Aerospace trajectory optimization",
     {"aerospace","trajectory","orbit","satellite","launch","reentry","guidance","space"}),
    (23, "vehicle_perception",     "Autonomous vehicle perception",
     {"autonomous","vehicle","perception","lidar","camera","detection","object","scene"}),
    (24, "vehicle_planning",       "Autonomous vehicle planning",
     {"autonomous","vehicle","planning","path","motion","navigation","driving","lane"}),
    (25, "swarm_intelligence",     "Swarm intelligence systems",
     {"swarm","intelligence","ant","particle","colony","optimization","collective","emergent"}),
    (26, "digital_twin",           "Digital twin industrial modeling",
     {"digital","twin","industrial","simulation","real","time","synchronization","virtual"}),
    (27, "manufacturing_opt",      "Manufacturing process optimization",
     {"manufacturing","process","optimization","production","quality","scheduling","yield"}),
    (28, "supply_chain",           "Supply chain network optimization",
     {"supply","chain","network","logistics","inventory","demand","distribution","optimization"}),
    (29, "logistics_routing",      "Logistics route optimization",
     {"logistics","route","routing","vehicle","delivery","tsp","vrp","schedule"}),
    (30, "warehouse_automation",   "Warehouse automation",
     {"warehouse","automation","picking","sorting","conveyor","agv","inventory","storage"}),
    (31, "smart_city_traffic",     "Smart city traffic modeling",
     {"smart","city","traffic","congestion","flow","intersection","signal","urban"}),
    (32, "infrastructure_stress",  "Infrastructure stress prediction",
     {"infrastructure","stress","bridge","road","load","failure","aging","maintenance"}),
    (33, "seismic_risk",           "Seismic risk modeling",
     {"seismic","earthquake","risk","ground","motion","fault","hazard","acceleration"}),
    (34, "climate_simulation",     "Climate simulation acceleration",
     {"climate","simulation","atmosphere","ocean","carbon","temperature","forcing","feedback"}),
    (35, "weather_forecasting",    "Weather forecasting",
     {"weather","forecast","atmospheric","precipitation","wind","pressure","humidity"}),
    (36, "flood_prediction",       "Flood prediction",
     {"flood","prediction","rainfall","runoff","river","basin","inundation","hydrological"}),
    (37, "wildfire_prediction",    "Wildfire prediction",
     {"wildfire","fire","spread","fuel","ignition","combustion","forest","prediction"}),
    (38, "carbon_cycle",           "Carbon cycle modeling",
     {"carbon","cycle","co2","sequestration","emission","flux","atmosphere","sink"}),
    (39, "ocean_modeling",         "Ocean current modeling",
     {"ocean","current","circulation","salinity","temperature","wave","tide","marine"}),
    (40, "agricultural_yield",     "Agricultural yield prediction",
     {"agricultural","yield","crop","soil","irrigation","harvest","growth","field"}),
    (41, "precision_agriculture",  "Precision agriculture optimization",
     {"precision","agriculture","sensor","drone","fertilizer","field","map","ndvi"}),
    (42, "genomic_modeling",       "Genomic sequence modeling",
     {"genomic","genome","sequence","dna","rna","gene","variant","mutation","snp"}),
    (43, "gene_regulatory",        "Gene regulatory network inference",
     {"gene","regulatory","network","expression","transcription","promoter","inference","grn"}),
    (44, "protein_structure",      "Protein structure prediction",
     {"protein","structure","folding","amino","acid","residue","conformation","alphafold"}),
    (45, "molecular_docking",      "Molecular docking prediction",
     {"molecular","docking","binding","ligand","receptor","affinity","pose","drug"}),
    (46, "drug_generation",        "Drug candidate generation",
     {"drug","candidate","generation","molecule","synthesis","activity","scaffold","lead"}),
    (47, "biomarker_discovery",    "Biomarker discovery",
     {"biomarker","discovery","disease","diagnosis","marker","clinical","omics","proteomics"}),
    (48, "personalized_treatment", "Personalized treatment modeling",
     {"personalized","treatment","patient","clinical","therapy","outcome","response","precision"}),
    (49, "radiology_imaging",      "Radiology image interpretation",
     {"radiology","image","mri","ct","xray","segmentation","detection","tumor","lesion"}),
    (50, "pathology_classification","Pathology classification",
     {"pathology","classification","tissue","biopsy","histology","slide","cancer","cell"}),
    (51, "ecg_eeg_anomaly",        "ECG/EEG anomaly detection",
     {"ecg","eeg","anomaly","detection","cardiac","arrhythmia","brainwave","seizure"}),
    (52, "icu_risk",               "ICU risk prediction",
     {"icu","intensive","care","risk","mortality","sepsis","prediction","patient","vital"}),
    (53, "hospital_operations",    "Hospital operations optimization",
     {"hospital","operations","scheduling","staffing","bed","resource","admission"}),
    (54, "epidemiological",        "Epidemiological forecasting",
     {"epidemiological","epidemic","sir","transmission","infection","reproduction","vaccination"}),
    (55, "wearable_health",        "Wearable health data integration",
     {"wearable","health","sensor","activity","heart","rate","accelerometer","monitoring"}),
    (56, "neural_decoding",        "Neural decoding of brain activity",
     {"neural","decoding","brain","activity","spike","neuron","cortex","electrode"}),
    (57, "brain_computer_interface","Brain-computer interface modeling",
     {"brain","computer","interface","bci","eeg","neural","control","prosthetic"}),
    (58, "cognitive_state",        "Cognitive state estimation",
     {"cognitive","state","attention","workload","fatigue","mental","arousal","fmri"}),
    (59, "population_dynamics",    "Population dynamics modeling",
     {"population","dynamics","growth","predator","prey","lotka","volterra","ecology"}),
    (60, "ecological_modeling",    "Ecological system modeling",
     {"ecological","ecosystem","species","biodiversity","habitat","food","web","niche"}),
    (61, "financial_forecasting",  "Financial time-series forecasting",
     {"financial","time","series","forecasting","stock","return","volatility","price"}),
    (62, "hft_optimization",       "High-frequency trading optimization",
     {"high","frequency","trading","hft","latency","order","market","microstructure"}),
    (63, "options_pricing",        "Options pricing approximation",
     {"options","pricing","black","scholes","volatility","derivative","greeks","hedge"}),
    (64, "portfolio_optimization", "Portfolio optimization",
     {"portfolio","optimization","allocation","risk","return","markowitz","sharpe","efficient"}),
    (65, "systemic_risk",          "Systemic risk modeling",
     {"systemic","risk","contagion","network","interconnected","shock","cascade","financial"}),
    (66, "credit_scoring",         "Credit scoring",
     {"credit","scoring","default","loan","probability","rating","debt"}),
    (67, "fraud_detection",        "Fraud detection",
     {"fraud","detection","anomaly","transaction","suspicious","imbalanced","classification"}),
    (68, "insurance_risk",         "Insurance risk modeling",
     {"insurance","risk","premium","actuarial","claim","loss","reserve","mortality"}),
    (69, "actuarial_forecasting",  "Actuarial forecasting",
     {"actuarial","mortality","survival","life","table","probability","insurance","pension"}),
    (70, "macroeconomic_forecasting","Macroeconomic forecasting",
     {"macroeconomic","gdp","inflation","unemployment","monetary","fiscal","forecast","var"}),
    (71, "commodity_markets",      "Commodity market prediction",
     {"commodity","market","oil","gas","gold","wheat","futures","price","supply"}),
    (72, "real_estate",            "Real estate valuation modeling",
     {"real","estate","property","valuation","housing","price","rental","market"}),
    (73, "auction_mechanisms",     "Auction mechanism optimization",
     {"auction","mechanism","bidding","pricing","market","design","equilibrium","revenue"}),
    (74, "behavioral_economics",   "Behavioral economics modeling",
     {"behavioral","economics","prospect","utility","bias","heuristic","decision","nudge"}),
    (75, "social_influence",       "Social influence propagation modeling",
     {"social","influence","network","propagation","diffusion","viral","cascade","opinion"}),
    (76, "misinformation",         "Misinformation diffusion modeling",
     {"misinformation","diffusion","fake","news","spread","rumor","detection","social"}),
    (77, "collective_decision",    "Collective decision simulation",
     {"collective","decision","voting","consensus","coordination","game","mechanism"}),
    (78, "multi_agent_game",       "Multi-agent game strategy learning",
     {"multi","agent","game","strategy","nash","equilibrium","reinforcement","learning"}),
    (79, "knowledge_graph",        "Knowledge graph reasoning",
     {"knowledge","graph","reasoning","entity","relation","embedding","kg","triple"}),
    (80, "large_language_model",   "Large language modeling",
     {"language","model","transformer","attention","token","llm","gpt","bert","embedding"}),
    (81, "code_synthesis",         "Code synthesis",
     {"code","synthesis","generation","program","function","ast","repair","completion"}),
    (82, "theorem_assistance",     "Automated theorem assistance",
     {"theorem","proof","formal","verification","logic","axiom","lean","coq","mathlib"}),
    (83, "literature_synthesis",   "Scientific literature synthesis",
     {"literature","synthesis","paper","abstract","citation","scientific","review","survey"}),
    (84, "hypothesis_generation",  "Hypothesis generation systems",
     {"hypothesis","generation","scientific","discovery","conjecture","experiment","test"}),
    (85, "legal_analysis",         "Legal contract analysis",
     {"legal","contract","clause","law","compliance","risk","obligation","party"}),
    (86, "regulatory_compliance",  "Regulatory compliance automation",
     {"regulatory","compliance","regulation","rule","standard","audit","requirement"}),
    (87, "cyber_intrusion",        "Cyber intrusion detection",
     {"cyber","intrusion","detection","ids","network","attack","malware","anomaly"}),
    (88, "network_anomaly",        "Network anomaly detection",
     {"network","anomaly","detection","traffic","packet","flow","intrusion","monitoring"}),
    (89, "threat_intelligence",    "Cyber threat intelligence synthesis",
     {"threat","intelligence","ioc","vulnerability","exploit","malware","ttps","cyber"}),
    (90, "military_strategy",      "Military strategy simulation",
     {"military","strategy","simulation","warfare","mission","planning","decision","combat"}),
    (91, "defense_systems",        "Autonomous defense systems",
     {"defense","autonomous","system","tracking","detection","interception","radar","sensor"}),
    (92, "satellite_imagery",      "Satellite imagery analysis",
     {"satellite","imagery","remote","sensing","land","cover","classification","sar"}),
    (93, "remote_sensing",         "Remote sensing analytics",
     {"remote","sensing","lidar","hyperspectral","multispectral","ndvi","vegetation"}),
    (94, "energy_demand",          "Energy demand forecasting",
     {"energy","demand","forecasting","load","consumption","building","peak","grid"}),
    (95, "water_resources",        "Water resource optimization",
     {"water","resource","optimization","irrigation","aquifer","basin","allocation","drought"}),
    (96, "urban_planning",         "Urban planning simulation",
     {"urban","planning","simulation","land","use","zoning","growth","infrastructure"}),
    (97, "construction_risk",      "Construction project risk modeling",
     {"construction","project","risk","schedule","cost","delay","building","management"}),
    (98, "smart_building",         "Smart building energy control",
     {"smart","building","energy","control","hvac","occupancy","sensor","automation"}),
    (99, "hr_workforce",           "Human resource workforce planning",
     {"human","resource","workforce","planning","talent","skill","hiring","retention"}),
    (100, "talent_matching",       "Talent matching optimization",
     {"talent","matching","job","skill","resume","candidate","recruitment","compatibility"}),
    (101, "education_personalization","Education personalization systems",
     {"education","personalization","learning","student","adaptive","curriculum","recommendation"}),
    (102, "curriculum_optimization","Curriculum optimization modeling",
     {"curriculum","optimization","course","learning","objective","sequence","mastery"}),
    (103, "student_dropout",       "Student dropout prediction",
     {"student","dropout","prediction","retention","academic","performance","engagement"}),
    (104, "recommendation_systems","Recommendation systems",
     {"recommendation","system","collaborative","filtering","content","user","item","matrix"}),
    (105, "dynamic_pricing",       "Dynamic pricing systems",
     {"dynamic","pricing","demand","revenue","management","yield","elasticity","optimization"}),
    (106, "advertising_optimization","Advertising optimization",
     {"advertising","optimization","bid","campaign","click","conversion","targeting","auction"}),
    (107, "market_segmentation",   "Market segmentation modeling",
     {"market","segmentation","clustering","customer","segment","demographic","behavior"}),
    (108, "customer_lifetime",     "Customer lifetime value prediction",
     {"customer","lifetime","value","clv","churn","retention","purchase","prediction"}),
    (109, "brand_sentiment",       "Brand sentiment monitoring",
     {"brand","sentiment","monitoring","social","media","opinion","review","reputation"}),
    (110, "content_moderation",    "Content moderation systems",
     {"content","moderation","toxic","harmful","classifier","safety","filter","detection"}),
    (111, "synthetic_data",        "Synthetic data generation",
     {"synthetic","data","generation","augmentation","privacy","gan","diffusion","anonymization"}),
    (112, "generative_image",      "Generative image modeling",
     {"generative","image","gan","diffusion","stable","vae","synthesis","generation"}),
    (113, "generative_video",      "Generative video synthesis",
     {"generative","video","synthesis","temporal","frame","motion","diffusion","generation"}),
    (114, "generative_music",      "Generative music composition",
     {"generative","music","composition","audio","melody","rhythm","harmony","synthesis"}),
    (115, "architectural_design",  "Architectural generative design",
     {"architectural","design","generative","floor","plan","layout","building","spatial"}),
    (116, "product_design",        "Product design optimization",
     {"product","design","optimization","cad","geometry","shape","topology","manufacturing"}),
    (117, "fashion_forecasting",   "Fashion trend forecasting",
     {"fashion","trend","forecasting","style","clothing","retail","season","demand"}),
    (118, "retail_inventory",      "Retail inventory forecasting",
     {"retail","inventory","forecasting","demand","stock","replenishment","supply","sku"}),
    (119, "telecom_fraud",         "Fraud detection in telecom",
     {"telecom","fraud","detection","call","roaming","billing","subscription","anomaly"}),
    (120, "network_bandwidth",     "Network bandwidth optimization",
     {"network","bandwidth","optimization","routing","congestion","latency","qos","traffic"}),
    (121, "edge_learning",         "Edge device adaptive learning",
     {"edge","device","adaptive","learning","iot","embedded","inference","compression"}),
    (122, "federated_learning",    "Federated learning across institutions",
     {"federated","learning","privacy","distributed","gradient","aggregation","client","server"}),
    (123, "autonomous_experiments","Autonomous scientific experiment design",
     {"autonomous","experiment","design","active","bayesian","optimization","scientific"}),
    (124, "lab_robotics",          "Laboratory robotics automation",
     {"laboratory","robotics","automation","pipetting","sample","high","throughput","assay"}),
    (125, "chemical_synthesis",    "Autonomous chemical synthesis planning",
     {"chemical","synthesis","planning","reaction","pathway","retrosynthesis","molecule","route"}),
    (126, "materials_failure",     "Materials stress failure forecasting",
     {"materials","stress","failure","fatigue","fracture","crack","deformation","creep"}),
    (127, "mining_estimation",     "Mining resource estimation",
     {"mining","resource","estimation","ore","reserve","grade","geostatistics","kriging"}),
    (128, "oil_gas_reservoir",     "Oil and gas reservoir modeling",
     {"oil","gas","reservoir","modeling","porous","flow","permeability","porosity","simulation"}),
    (129, "transportation_network","Transportation network optimization",
     {"transportation","network","optimization","flow","capacity","routing","assignment"}),
    (130, "maritime_routing",      "Maritime route optimization",
     {"maritime","route","optimization","ship","vessel","port","navigation","sea"}),
    (131, "aviation_scheduling",   "Aviation scheduling optimization",
     {"aviation","scheduling","optimization","flight","crew","airport","gate","delay"}),
    (132, "disaster_response",     "Disaster response allocation modeling",
     {"disaster","response","allocation","humanitarian","relief","emergency","logistics"}),
    (133, "humanitarian_logistics","Humanitarian logistics planning",
     {"humanitarian","logistics","planning","aid","distribution","crisis","supply","relief"}),
    (134, "crowd_behavior",        "Crowd behavior prediction",
     {"crowd","behavior","prediction","pedestrian","evacuation","flow","density","panic"}),
    (135, "security_surveillance", "Security surveillance anomaly detection",
     {"security","surveillance","anomaly","detection","camera","video","behavior","threat"}),
    (136, "identity_verification", "Identity verification systems",
     {"identity","verification","authentication","document","face","recognition","fraud"}),
    (137, "biometric_auth",        "Biometric authentication",
     {"biometric","authentication","fingerprint","face","iris","gait","voice","recognition"}),
    (138, "emotion_recognition",   "Emotion recognition systems",
     {"emotion","recognition","facial","expression","sentiment","affective","valence","arousal"}),
    (139, "speech_recognition",    "Speech recognition",
     {"speech","recognition","asr","acoustic","phoneme","transcription","language","model"}),
    (140, "speaker_identification","Speaker identification",
     {"speaker","identification","verification","voice","diarization","embedding","x-vector"}),
    (141, "language_translation",  "Language translation",
     {"language","translation","machine","neural","mt","beam","attention","multilingual"}),
    (142, "multimodal_reasoning",  "Multimodal reasoning (vision+language+audio)",
     {"multimodal","reasoning","vision","language","audio","fusion","alignment","cross"}),
    (143, "autonomous_research",   "Autonomous research agents",
     {"autonomous","research","agent","scientific","experiment","hypothesis","discovery"}),
    (144, "multimodal_memory",     "Multi-modal memory systems",
     {"multimodal","memory","retrieval","episodic","semantic","storage","context","recall"}),
    (145, "transfer_learning",     "Cross-domain transfer learning",
     {"transfer","learning","domain","adaptation","fine","tuning","pretrain","generalization"}),
    (146, "self_improving_rl",     "Self-improving reinforcement learning",
     {"self","improving","reinforcement","learning","reward","policy","exploration","agent"}),
    (147, "meta_learning",         "Meta-learning optimization systems",
     {"meta","learning","optimization","few","shot","maml","hyperparameter","inner","outer"}),
    (148, "hierarchical_decisions","Hierarchical decision architectures",
     {"hierarchical","decision","architecture","planning","abstract","subgoal","option","hrl"}),
    (149, "global_systems_modeling","Global systems modeling",
     {"global","systems","modeling","physical","economic","social","integrated","agent","world"}),
]

# Fast lookup structures
_ID_TO_ENTRY: Dict[str, Tuple] = {entry[1]: entry for entry in DOMAIN_MANIFEST}
_IDX_TO_ENTRY: Dict[int, Tuple] = {entry[0]: entry for entry in DOMAIN_MANIFEST}
_ALL_IDS: List[str] = [entry[1] for entry in DOMAIN_MANIFEST]


class DomainRegistry:
    """
    Manages the 150-domain expansion map for the FICUTS system.

    Responsibilities:
      - Classify incoming text to the best-matching domain (keyword overlap)
      - Activate dormant domains → allocate HDV subspace dims
      - Map domain names to UnifiedTensorNetwork mode head indices
      - Persist activation state across sessions
    """

    def __init__(
        self,
        persist_path: str = "tensor/data/active_domains.json",
        hdv_system=None,
    ):
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Load previously activated domains
        self._active: Set[str] = set()
        if self.persist_path.exists():
            try:
                saved = json.loads(self.persist_path.read_text())
                self._active = set(saved.get("active", []))
            except Exception:
                pass

        # If hdv_system provided, pre-register already-active domains
        if hdv_system is not None:
            for domain_id in list(self._active):
                # Ensure domain mask exists (cheap no-op if already registered)
                hdv_system.structural_encode("", domain_id)

    # ── Classification ─────────────────────────────────────────────────────────

    def classify_domain(self, text: str, top_k: int = 1) -> List[str]:
        """
        Classify text to the best-matching domain(s) by keyword overlap.

        Tokenises text (lowercase, split on non-alpha), counts how many
        keywords from each domain appear in the text, returns the top_k
        domain IDs ordered by descending match score.

        Returns ["unknown"] if no domain has any keyword match.
        """
        tokens = set(re.split(r"[^a-z0-9]+", text.lower())) - {"", "the", "a", "of", "in", "for", "and", "to"}

        scores: Dict[str, int] = {}
        for _, domain_id, _, keywords in DOMAIN_MANIFEST:
            score = len(keywords & tokens)
            if score > 0:
                scores[domain_id] = score

        if not scores:
            return ["unknown"]
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        return ranked[:top_k]

    def best_domain(self, text: str) -> str:
        """Return single best-matching domain ID for text."""
        return self.classify_domain(text, top_k=1)[0]

    # ── Activation ─────────────────────────────────────────────────────────────

    def activate_domain(self, domain_id: str, hdv_system=None) -> int:
        """
        Activate a domain: allocate HDV subspace dimensions and persist.

        Args:
            domain_id:  one of the 150 domain IDs (snake_case)
            hdv_system: IntegratedHDVSystem — if provided, registers domain mask

        Returns: mode head index for this domain (0..149), or -1 if unknown.
        """
        if domain_id not in _ID_TO_ENTRY:
            return -1

        with self._lock:
            if domain_id not in self._active:
                self._active.add(domain_id)

                # Allocate HDV subspace (auto-creates domain mask)
                if hdv_system is not None:
                    hdv_system.structural_encode(f"domain {domain_id}", domain_id)

                self._save()

        return _ID_TO_ENTRY[domain_id][0]

    def activate_for_text(self, text: str, hdv_system=None) -> Tuple[str, int]:
        """
        Classify text, activate matching domain, return (domain_id, head_idx).
        """
        domain_id = self.best_domain(text)
        if domain_id == "unknown":
            return "unknown", -1
        head_idx = self.activate_domain(domain_id, hdv_system)
        return domain_id, head_idx

    # ── Queries ────────────────────────────────────────────────────────────────

    def active_domains(self) -> List[str]:
        with self._lock:
            return sorted(self._active)

    def inactive_domains(self) -> List[str]:
        with self._lock:
            return [d for d in _ALL_IDS if d not in self._active]

    def is_active(self, domain_id: str) -> bool:
        with self._lock:
            return domain_id in self._active

    def get_head_idx(self, domain_id: str) -> int:
        entry = _ID_TO_ENTRY.get(domain_id)
        return entry[0] if entry else -1

    def get_display_name(self, domain_id: str) -> str:
        entry = _ID_TO_ENTRY.get(domain_id)
        return entry[2] if entry else domain_id

    def get_keywords(self, domain_id: str) -> Set[str]:
        entry = _ID_TO_ENTRY.get(domain_id)
        return entry[3] if entry else set()

    def n_active(self) -> int:
        with self._lock:
            return len(self._active)

    def n_inactive(self) -> int:
        return 150 - self.n_active()

    # ── Status ─────────────────────────────────────────────────────────────────

    def status(self) -> Dict:
        active = self.active_domains()
        return {
            "total_domains": 150,
            "active": len(active),
            "inactive": 150 - len(active),
            "active_list": active[:20],          # first 20 for display
            "coverage_pct": round(100.0 * len(active) / 150, 1),
        }

    def _save(self):
        """Persist active set (caller must hold self._lock)."""
        try:
            self.persist_path.write_text(
                json.dumps({"active": sorted(self._active)}, indent=2)
            )
        except Exception:
            pass
