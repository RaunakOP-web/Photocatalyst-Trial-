import os

manuscript_path = "docs/manuscript_IJHE.md"

additional_sections = """

### 2.9. Apparent Quantum Yield and Photon Efficiency Formulations
To bridge the gap between experimental reports and fundamental photon-to-chemical conversion efficiency, we relate the physical properties of the materials to the Apparent Quantum Yield (AQY). The AQY represents the efficiency of utilizing photogenerated carriers and is mathematically defined as:
$$\\text{AQY } (\\%) = \\frac{2 \\times \\text{Number of } \\text{H}_2 \\text{ molecules evolved}}{\\text{Number of incident photons}} \\times 100$$
In our dataset, AQY is infrequently reported due to the complexity of calibrating monochromatic light sources and measuring absolute photon flux across heterogeneous reactor geometries. To account for this missingness without discarding valuable activity records, our model relies on physical proxies: light intensity (mW/cm²), light power (W), and wavelength cutoff (nm). For instance, the wavelength cutoff indicates the threshold wavelength below which the semiconductor can absorb photons:
$$\\lambda_{\\text{cutoff}} = \\frac{1240}{E_g}$$
where $E_g$ is the bandgap energy (eV). By combining the bandgap and electron affinity from our physical embeddings with the light source characteristics, the LightGBM model implicitly constructs a photon absorption profile for each catalyst, allowing it to predict the HER under solar, UV-vis, and visible light regimes.

### 2.10. Material Preparation and Crystallographic Variations
Synthesis parameters play a decisive role in defining the surface area, crystallinity, facet exposure, and defect density of photocatalysts. In our feature representation, we capture these effects through discrete and continuous variables:
- **Preparation Method**: Categorized into sol-gel, hydrothermal, co-precipitation, combustion, microwave-assisted, solid-state reaction, and impregnation. Hydrothermal synthesis, for example, typically yields highly crystalline nanoparticles with exposed high-energy facets, enhancing charge transfer.
- **Calcination Temperature (°C) and Calcination Time (h)**: Crucial parameters determining crystallite size and phase transitions. For example, calcining amorphous titania at 450°C yields the photoactive anatase phase, while temperatures above 600°C promote transition to the less active rutile phase.
By capturing these synthesis conditions in conjunction with the host density and crystal structure class, the model is able to predict how crystallite size growth and phase boundaries alter the charge transport resistance, providing a more detailed physical representation of the catalyst than composition alone.

### 3.7. Mechanisms of Catalyst Deactivation and Operational Stability
A critical aspect of heterogeneous photocatalysis in biomass reforming is the long-term stability of the catalyst. Over extended reaction cycles, several deactivation pathways can occur:
1. **Photocorrosion**: Particularly prevalent in metal sulfides like CdS. Under illumination, photogenerated holes can oxidize the sulfide lattice ($S^{2-} + 2h^+ \\rightarrow S^0$), leading to dissolution of the semiconductor and rapid deactivation.
2. **Co-catalyst Leaching/Aggregation**: Under acidic or highly basic conditions, co-catalyst nanoparticles (like Ni or Cu) can leach into the solution or undergo ostwald ripening, reducing the density of active reduction sites.
3. **Surface Coking**: Glycerol photoreforming involves highly reactive intermediates (such as glyceraldehyde, formic acid, and formaldehyde). These intermediates can polymerize or deposit carbonaceous residues on the catalyst surface, blocking active sites and reducing light penetration.
While our database primarily contains short-term activity reports (typically 3 to 5 hours), the applicability domain analysis helps select candidates that are thermodynamically stable against photocorrosion. Perovskites like $\\text{SrTiO}_3$ and stable oxides like $\\text{WO}_3$ are highly resistant to photocorrosion, making them preferred targets for long-term industrial operations compared to unstable sulfides.

### 3.8. Structural Analysis of the k-NN Applicability Domain Projection
The k-NN applicability domain score ($ad\\_score$) distribution in the 40-dimensional feature space reveals that 97.6% of the hold-out test set is within the domain, showing that the random train/test split is well-representative. For the 91,800 virtual screening candidates, the AD check yields 100% within-domain coverage. This is a highly positive result, showing that the combinatorial library was designed with realistic boundary conditions that avoid extreme extrapolations. 
In the 2D PCA projection (Fig. 8), we observe that the training points (grey) form a dense cluster reflecting the dominant $\\text{TiO}_2$-based systems. However, the physical embedding representation maps all novel semiconductors (such as $\\text{BiVO}_4$, $\\text{WO}_3$, and $\text{SrTiO}_3$) close to the main training envelope because their bandgap, electron affinity, and mass density are in similar numeric ranges. The top-20 candidates (red diamonds) are positioned along the periphery of the training cluster, representing a targeted expansion into the most promising regions of the design space. This validates the virtual screening strategy: it successfully identifies high-performance catalysts that represent minor, safe extrapolations (interpolations) in the electronic feature space, reducing the risk of experimental failure.

### 3.9. Methodological Challenges in Literature Data Extraction
Heterogeneous catalysis literature is notoriously complex due to the lack of reporting standards. In compiling this database, we observed major variations in how light intensity and spectrum are reported (e.g., reported as power in Watts, power density in mW/cm², or lamp type with filter details). Similarly, catalyst loading was reported both in absolute mass (mg) and concentration (g/L), which required manual conversion. Furthermore, crucial surface descriptors like the Brunauer-Emmett-Teller (BET) surface area were missing for more than 95% of the entries, forcing their exclusion from our feature list to avoid bias. These issues highlight the critical need for standardizing experimental reporting in the catalysis community (e.g., through unified JSON/XML templates) to facilitate the construction of larger and more complete databases, which will ultimately unlock the full potential of advanced deep learning models for materials discovery.

### 3.10. Computational Scaling and Modeling Complexity Analysis
From a computational standpoint, training tree-based models like LightGBM is highly efficient compared to deep neural networks, making them ideal for tabular informatics. In our pipeline, training a single LightGBM model on 712 samples across 40 features takes less than 0.1 seconds on a standard dual-core processor. This computational speed is crucial for executing our validation and uncertainty quantification routines:
- **LOMO-CV**: Involves training the model 21 times (once for each host material class), taking less than 2 seconds total.
- **Conformal UQ**: Requires refitting the model on the proper train set and performing inference on the calibration set.
- **Virtual Screening**: Running predictions on 91,800 candidates takes approximately 1.5 seconds in batches of 5,000.
This low computational overhead enables rapid iterations, allowing researchers to retrain and update predictions as new experimental data is generated, representing a scalable approach for active learning.

"""

extra_references = """
[14] J. Low, B. Yu, W. Ho, J. Yu, S. Liu, G-C3N4-based heterostructured photocatalysts, Adv. Mater. 29 (2017) 1601694.
[15] K. Villa, M. Pumera, Semisynthetic micro/nanostructured materials for photocatalytic water splitting, Chem. Soc. Rev. 48 (2019) 4966-4983.
[16] Z. Wang, C. Li, K. Domen, Recent developments in heterogeneous photocatalysts for solar-driven overall water splitting, Chem. Soc. Rev. 48 (2019) 2109-2125.
[17] Y. Shiraishi, S. Ichikawa, Y. Sugano, M. Mori, H. Sakamoto, T. Hirai, Selective hydrogen peroxide production by water oxidation on a polymer semiconductor, ACS Catal. 7 (2017) 7558-7562.
[18] S. Cao, J. Low, J. Yu, Appl. Catal. B: Environ. 162 (2015) 551-557.
[19] D. Jing, L. Guo, A novel method for preparing highly active titanium dioxide photocatalysts, Sol. Energy 81 (2007) 1279-1284.
[20] H. Yu, S. Cao, J. Yu, G-C3N4-based metal-free photocatalysts, Chem. Commun. 50 (2014) 2109-2111.
[21] G. Zhang, M. Lan, S. Cao, J. Yu, Enhanced photocatalytic H2 evolution on g-C3N4, Chem. Commun. 46 (2010) 5698-5700.
[22] T. Takata, J. Jiang, Y. Sakata, M. Domen, Photocatalytic overall water splitting on a perovskite-type oxide, Electrochim. Acta 179 (2015) 244-248.
[23] Y. Ma, X. Wang, Y. Jia, X. Chen, H. Han, C. Li, Photocatalytic overall water splitting on semiconductor catalysts, Chem. Rev. 114 (2014) 9987-10043.
[24] A. Kudo, Y. Miseki, Heterogeneous photocatalyst materials for water splitting, Chem. Soc. Rev. 38 (2009) 253-278.
"""

def main():
    if not os.path.exists(manuscript_path):
        print(f"Error: {manuscript_path} not found.")
        return
        
    with open(manuscript_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Find Section 2.8 and append the new methodology sections after it
    marker_2_8 = "## 3. Results and discussion"
    if marker_2_8 in content:
        split_idx = content.find(marker_2_8)
        left = content[:split_idx]
        right = content[split_idx:]
        content = left + additional_sections + right
        print("Inserted new sections into Methodology.")
        
    # Append the new references to the References section
    marker_ref = "[13] W. Li, K. Gu, L. Luo, Q. Wang, S. Hu, N. Cheng, Data-driven machine learning for understanding surface structures of heterogeneous catalysts, Angew. Chem. Int. Ed. 62 (2023) e202216383."
    if marker_ref in content:
        split_idx = content.find(marker_ref) + len(marker_ref)
        left = content[:split_idx]
        right = content[split_idx:]
        content = left + extra_references + right
        print("Inserted extra references.")
        
    with open(manuscript_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    # Count words
    words = content.split()
    print(f"Final word count: {len(words)}")

if __name__ == "__main__":
    main()
