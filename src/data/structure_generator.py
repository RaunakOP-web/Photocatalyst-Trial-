import numpy as np
from pymatgen.core import Structure, Lattice, Composition

def make_cubic_prototype(elements, coordinates, a=4.0):
    lattice = Lattice.cubic(a)
    species = []
    coords = []
    for el, cc_list in zip(elements, coordinates):
        for cc in cc_list:
            species.append(el)
            coords.append(cc)
    return Structure(lattice, species, coords)

def get_anatase_tio2():
    lattice = Lattice.tetragonal(3.784, 9.515)
    species = ["Ti"] * 4 + ["O"] * 8
    coords = [
        [0, 0, 0], [0, 0.5, 0.25], [0.5, 0, 0.75], [0.5, 0.5, 0.5],
        [0, 0, 0.20], [0, 0, 0.80], [0, 0.5, 0.45], [0, 0.5, 0.05],
        [0.5, 0, 0.55], [0.5, 0, 0.95], [0.5, 0.5, 0.30], [0.5, 0.5, 0.70]
    ]
    return Structure(lattice, species, coords)

def get_wurtzite_zno():
    lattice = Lattice.hexagonal(3.25, 5.20)
    species = ["Zn", "Zn", "O", "O"]
    coords = [
        [1/3, 2/3, 0], [2/3, 1/3, 0.5],
        [1/3, 2/3, 0.38], [2/3, 1/3, 0.88]
    ]
    return Structure(lattice, species, coords)

def get_zincblende_zns():
    lattice = Lattice.cubic(5.41)
    species = ["Zn"] * 4 + ["S"] * 4
    coords = [
        [0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0],
        [0.25,0.25,0.25], [0.25,0.75,0.75], [0.75,0.25,0.75], [0.75,0.75,0.25]
    ]
    return Structure(lattice, species, coords)

def get_fluorite_ceo2():
    lattice = Lattice.cubic(5.411)
    species = ["Ce"] * 4 + ["O"] * 8
    coords = [
        [0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0],
        [0.25,0.25,0.25], [0.25,0.25,0.75], [0.25,0.75,0.25], [0.25,0.75,0.75],
        [0.75,0.25,0.25], [0.75,0.25,0.75], [0.75,0.75,0.25], [0.75,0.75,0.75]
    ]
    return Structure(lattice, species, coords)

def get_perovskite_srtio3():
    lattice = Lattice.cubic(3.905)
    species = ["Sr", "Ti", "O", "O", "O"]
    coords = [
        [0, 0, 0], [0.5, 0.5, 0.5],
        [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]
    ]
    return Structure(lattice, species, coords)

def make_generic_structure(formula):
    """
    Parses chemical formula and places atoms in a reasonable cubic/orthorhombic
    lattice with cell volume proportional to formula weight / packing.
    """
    comp = Composition(formula)
    elements = list(comp.as_dict().keys())
    counts = [int(np.round(comp.as_dict()[el])) for el in elements]
    
    # Target volume estimated by atomic size
    total_atoms = sum(counts)
    vol_per_atom = 15.0  # approximate volume in cubic angstroms
    box_side = (total_atoms * vol_per_atom) ** (1/3)
    lattice = Lattice.cubic(box_side)
    
    # Place atoms in a grid
    grid_dim = int(np.ceil(total_atoms ** (1/3)))
    coords = []
    species = []
    
    idx = 0
    for el, cnt in zip(elements, counts):
        for _ in range(cnt):
            # Coordinates on a fraction grid
            xi = (idx % grid_dim) / grid_dim + 0.1 / grid_dim
            yi = ((idx // grid_dim) % grid_dim) / grid_dim + 0.1 / grid_dim
            zi = (idx // (grid_dim * grid_dim)) / grid_dim + 0.1 / grid_dim
            coords.append([xi, yi, zi])
            species.append(el)
            idx += 1
            
    return Structure(lattice, species, coords)

def formula_to_structure(formula):
    """
    Main route to convert formulas to pymatgen structure templates.
    """
    f_lower = formula.lower().strip().replace(" ", "")
    
    # Clean formula synonyms
    if f_lower in ("tio2", "tio3"):
        return get_anatase_tio2()
    elif f_lower == "zno":
        return get_wurtzite_zno()
    elif f_lower in ("zns", "hex-zns"):
        return get_zincblende_zns()
    elif f_lower in ("cds", "hex-cds"):
        return get_zincblende_zns().replace_species({"Zn": "Cd"})
    elif f_lower == "ceo2":
        return get_fluorite_ceo2()
    elif f_lower == "pbtio3":
        return get_perovskite_srtio3().replace_species({"Sr": "Pb"})
    elif f_lower == "srtio3":
        return get_perovskite_srtio3()
    elif f_lower == "cuo":
        # simple rocksalt style for CuO
        lattice = Lattice.cubic(4.2)
        species = ["Cu"] * 4 + ["O"] * 4
        coords = [
            [0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0],
            [0.5,0.5,0.5], [0.5,0,0], [0,0.5,0], [0,0,0.5]
        ]
        return Structure(lattice, species, coords)
    elif f_lower == "wo3":
        lattice = Lattice.cubic(3.8)
        species = ["W", "O", "O", "O"]
        coords = [[0,0,0], [0.5,0,0], [0,0.5,0], [0,0,0.5]]
        return Structure(lattice, species, coords)
    elif f_lower == "g-c3n4":
        # Simplified heptazine unit cell representation
        lattice = Lattice.hexagonal(4.7, 6.4)
        species = ["C", "C", "N", "N", "N", "N"]
        coords = [
            [1/3, 2/3, 0], [2/3, 1/3, 0],
            [0,0,0], [1/3, 2/3, 0.5], [2/3, 1/3, 0.5], [0, 0, 0.5]
        ]
        return Structure(lattice, species, coords)
    elif f_lower == "zno-zns":
        # Solid solution model
        s = get_zincblende_zns()
        s.replace_species({"Zn": "Zn", "S": "O"})
        # replace half of O with S
        species = list(s.species)
        species[2] = Composition("S").elements[0]
        species[3] = Composition("S").elements[0]
        return Structure(s.lattice, species, s.frac_coords)
    elif f_lower == "cd1-xznxs":
        # ZnS lattice, but substitute Cd for some Zn
        s = get_zincblende_zns()
        species = list(s.species)
        species[0] = Composition("Cd").elements[0]
        species[1] = Composition("Cd").elements[0]
        return Structure(s.lattice, species, s.frac_coords)
    
    # Generic backup
    try:
        # Try to parse as valid formula
        cleaned_f = formula.split("-")[0].split("/")[0]  # E.g. zno-zns -> zno
        return make_generic_structure(cleaned_f)
    except Exception:
        # Solid fallback
        return get_anatase_tio2()
