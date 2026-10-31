"""Wartości kontrolne z publikacji.

Każda liczba ma podane źródło. Z tych samych stałych korzystają testy krzyżowe z qkd
(tests/crosscheck) i testy własnego modułu qkdsim.analysis.
"""

# Parametry GYS: Ma, Qi, Zhao, Lo 2005, Tabela I; f_EC z tekstu artykułu (docs/model.md, rozdz. 3).
GYS_ALPHA_DB_PER_KM = 0.21
GYS_ETA_BOB = 0.045
GYS_Y0_PER_PULSE = 1.7e-6
GYS_E_DETECTOR = 0.033
GYS_F_EC = 1.22
E0 = 0.5
Q_SIFT = 0.5

# Ma et al. 2005, rys. 2: maksymalny zasięg protokołu decoy [km].
L_MAX_INFINITE_DECOY_KM = 142.05
L_MAX_VACUUM_WEAK_KM = 140.55
VACUUM_WEAK_MU = 0.48
VACUUM_WEAK_NU = 0.05
L_MAX_TOLERANCE_KM = 0.5

# Ma et al. 2005, rozdz. 3.1, wzór (12): optymalna intensywność sygnału.
MU_OPT_F_EC = 0.48
MU_OPT_SHANNON = 0.54
MU_OPT_TOLERANCE = 5e-3  # zaokrąglenie do dwóch miejsc w artykule

# Ma et al. 2005, rys. 1: odchylenie Y1^L i e1^U od granicy nieskończenie wielu stanów decoy
# przy 40 km i nu/mu = 25% [%].
DEVIATION_40KM_Y1_PERCENT = 3.5
DEVIATION_40KM_E1_PERCENT = 16.8
DEVIATION_TOLERANCE_PERCENT = 0.05  # zaokrąglenie do jednego miejsca w artykule

# GLLP bez stanów decoy: własne przeliczenie (≈ 40 km), potwierdzone modelem
# zysku qkd 0.2.0 (40,20 km). Artykuł nie podaje tej liczby wprost.
L_MAX_GLLP_NO_DECOY_KM = 40.2

# Lydersen et al. 2010, podpis rysunku „Detector control” (Clavis2): moc impulsu wyzwalającego,
# przy której detektor 0 / 1 nigdy nie klika (P_never) i zawsze klika (P_always) [W].
LYDERSEN_P_NEVER_W = (647e-6, 697e-6)
LYDERSEN_P_ALWAYS_W = (808e-6, 932e-6)

# Granice analityczne.
SHOR_PRESKILL_QBER = 0.110028  # pierwiastek 1 - 2h(Q) = 0
FUCHS_CROSSING_D = 0.146447  # (1 - 1/sqrt(2)) / 2
ANALYTIC_TOLERANCE = 1e-6

# Zhao et al. 2008, Tabela I(c): wyniki z danych Tabeli I(a)–(b) (system ID-500, μ = 0,1).
ZHAO_BALANCING_P_A = 0.230  # p_A równoważące zera i jedynki Boba
ZHAO_QBER = 0.0568
ZHAO_K_UPPER_BITS = 1131
ZHAO_K_LOWER_BITS = 1297
ZHAO_ROUNDING = 1.0  # artykuł podaje K w pełnych bitach

# Gisin et al. 2006, podpis rys. 7: informacja Ewy przy 0,1 fotonu w odbitej sondzie [bit].
GISIN_MU = 0.1
GISIN_INFORMATION_COHERENT = 0.135
GISIN_INFORMATION_RANDOM_PHASE = 0.095
GISIN_ROUNDING = 5e-4

# NIST SP 800-22 rev. 1a, przykłady z rozdz. 2.1.4 (monobit) i 2.3.4 (runs), n = 10.
NIST_MONOBIT_EXAMPLE = ("1011010101", 0.527089)
NIST_RUNS_EXAMPLE = ("1001101011", 0.147232)
NIST_PVALUE_DIGITS = 5e-7

# Lim et al. 2014 (arXiv:1311.7129), tekst pod rys. 1: przy n_X = 10⁴ klucz do 135 km, a przy
# 100 km stopa dla n_X = 10⁹ to „about 1.75 times” stopy dla 10⁷.
LIM_MAX_DISTANCE_KM_1E4 = 135.0
LIM_RATIO_100KM = 1.75
LIM_RATIO_TOLERANCE = 0.05  # „about” w tekście

# Lo, Curty, Qi 2012: przekaźnik pośrodku toleruje „more than 40dB (i.e., 200km)” strat
# (parametry z podpisu rys. 2).
LCQ12_MIN_RANGE_KM = 200.0

# Device-independent QKD: próg sprawności detekcji dla stanu maksymalnie splątanego przy Q = 0
# i S = 2√2, brak kliknięcia czytany jako wynik (Pironio i in., New J. Phys. 11, 045021 (2009),
# podpis rys. 3, za dokumentacją qkd 0.2.0: DI_ETA_SINGLET). Podany do 0,1 punktu procentowego.
PIRONIO_DI_EFFICIENCY = 0.924
PIRONIO_ROUNDING = 2e-3
