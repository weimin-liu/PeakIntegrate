from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
from PeakIntegrate.src.postprocess import *


# ---- Mathematical Models ----
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def double_gauss(x, A1, mu1, sigma1, A2, mu2, sigma2):
    return gauss(x, A1, mu1, sigma1) + gauss(x, A2, mu2, sigma2)


# ---- Load Experiment ----
try:
    with open("../../experiment.pkl", "rb") as f:
        exp = pickle.load(f)
except FileNotFoundError:
    print("Error: 'experiment.pkl' not found.")
    exit()

target_cmpds = [
    'C46-GDGT', 'brGDGT_IIIa_0', 'brGDGT_IIIa_1', 'brGDGT_IIIa_2',
    'brGDGT_IIIb', 'brGDGT_IIIc', 'brGDGT_IIa_0', 'brGDGT_IIa_1',
    'brGDGT_IIb', 'brGDGT_IIc', 'brGDGT_Ia', 'brGDGT_Ib', 'brGDGT_Ic'
]

all_results = {}

for cmpd in target_cmpds:
    print(f"Processing {cmpd}...")

    try:
        rtmin, rtmax, rtmed = exp.get_rt(cmpd).values()
    except Exception as e:
        print(f"  Skipping {cmpd}: Could not get RT ({e})")
        continue

    cmpd_results = {}

    for sample_name, chrom_obj in exp.chromatograms.items():

        # 1. Efficient EIC Extraction
        matching_eic = next((eic for eic in chrom_obj.eics if cmpd.startswith(eic.name)), None)
        if matching_eic is None:
            cmpd_results[sample_name] = np.nan
            continue

        if sample_name == 'AEGIS-139' and cmpd=='brGDGT_IIa_1':
            print('yeah')

        rt = np.asarray(matching_eic.shifted_rt, dtype=float)
        intensity = np.asarray(matching_eic.intensity, dtype=float)

        # 2. Masking & Data Cleaning
        mask = (rt > rtmin) & (rt < rtmax) & np.isfinite(rt) & np.isfinite(intensity)
        x = rt[mask]
        y = intensity[mask]

        # 3. Handle Sparse Data
        if len(x) < 11:
            cmpd_results[sample_name] = np.nan
            continue

        y_s = savgol_filter(y, window_length=11, polyorder=3)
        max_intensity = y_s.max()

        if max_intensity <= 0:
            cmpd_results[sample_name] = 0.0
            continue

        # 4. COUNT THE PEAKS
        # Prominence of 5% means it ignores noise bumps smaller than 5% of the highest peak
        peaks_indices, _ = find_peaks(y_s, prominence=max_intensity * 0.05)
        num_peaks = len(peaks_indices)

        area_main = np.nan

        # ==========================================
        # SCENARIO A: ONLY ONE PEAK FOUND
        # ==========================================
        if num_peaks == 1:
            apex_rt = x[peaks_indices[0]]
            apex_int = y_s[peaks_indices[0]]

            try:
                popt1, _ = curve_fit(
                    gauss, x, y_s,
                    p0=[apex_int, apex_rt, 5],
                    bounds=([0, rtmin, 1], [np.inf, rtmax, 30]),
                    maxfev=10000
                )
                A, mu, sigma = popt1
                area_main = A * sigma * np.sqrt(2 * np.pi)
            except Exception:
                pass

                # ==========================================
        # SCENARIO B: TWO (OR MORE) PEAKS FOUND
        # ==========================================
        elif num_peaks >= 2:
            # Get the top 2 highest peaks if it found more than 2
            top_2_indices = sorted(peaks_indices, key=lambda i: y_s[i], reverse=True)[:2]

            rt1, rt2 = x[top_2_indices[0]], x[top_2_indices[1]]
            int1, int2 = y_s[top_2_indices[0]], y_s[top_2_indices[1]]

            try:
                popt2, _ = curve_fit(
                    double_gauss, x, y_s,
                    # We now have the exact actual locations of the two maxima to give the solver!
                    p0=[int1, rt1, 5, int2, rt2, 5],
                    bounds=(
                        [0, rtmin, 1, 0, rtmin, 1],
                        [np.inf, rtmax, 30, np.inf, rtmax, 30]
                    ),
                    maxfev=10000
                )
                A1, mu1, sigma1, A2, mu2, sigma2 = popt2

                # Pick the peak closest to expected rtmed
                if abs(mu1 - rtmed) < abs(mu2 - rtmed):
                    area_main = A1 * sigma1 * np.sqrt(2 * np.pi)
                else:
                    area_main = A2 * sigma2 * np.sqrt(2 * np.pi)
            except Exception:
                pass

        # If 0 peaks were found, it stays np.nan
        cmpd_results[sample_name] = area_main

    all_results[cmpd] = cmpd_results

# ---- Data Export ----
results_df = pd.DataFrame(all_results)
results_df.index.name = "Sample"
results_df = results_df.sort_index()
results_df.to_csv('results.csv')

print("\nProcessing complete. Results saved to 'results.csv'.")