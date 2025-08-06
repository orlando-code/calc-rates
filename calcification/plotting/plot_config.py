import seaborn as sns

SCENARIO_MAP = {
    "ssp126": "SSP 1-2.6",
    "ssp245": "SSP 2-4.5",
    "ssp370": "SSP 3-7.0",
    "ssp585": "SSP 5-8.5",
}

RATE_TYPE_MAPPING = {
    "mgCaCO3 cm-2d-1": r"$mg \, CaCO_3 \, cm^{-2} \, d^{-1}$",
    "mgCaCO3 g-1d-1": r"$mg \, CaCO_3 \, g^{-1} \, d^{-1}$",
    "delta mass d-1": r"$\Delta mass\, d^{-1}$",
    "mg d-1": r"$mg \, d^{-1}$",
    "m2 d-1": r"$m^2 \, d^{-1}$",
    "m d-1": r"$m \, d^{-1}$",
    "deltaSA d-1": r"$\Delta SA \, d^{-1}$",
}

# Using seaborn's vlag colormap (violet-lavender-amber-gold)
# This creates a diverging palette that goes from violet/blue to amber/gold
vlag_palette = sns.color_palette("colorblind", n_colors=5)
CG_COLOURS = {
    "Coral": vlag_palette[1],  # Violet/blue end of the spectrum
    "CCA": vlag_palette[4],  # Blue-ish color
    "Halimeda": vlag_palette[2],  # Middle/neutral color
    "Other algae": vlag_palette[0],  # Amber-ish color
    "Foraminifera": vlag_palette[3],  # Gold/red end of the spectrum
}
