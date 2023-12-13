"""
This file holds constants that I use throughout the simulation. 

I use a different form of units where the mass of a lithium 7 atom is taken as 1 kg. This changes the value of the bohr
magneton. Every other constant I use is the same.

The new units are derived by:
F=ma=uB*grad(V) -> m_unitless*1kg*a=uB*grad(V) -> 1kg*a=(uB/m_unitless)*grad(V) ->1kg*a=uB' *grad(V)
thus: uB' = uB/m_unitless
where: m_unitless=unitless number equal to mass of lithium in units of kg

where V is the magnetic potential energy of a lithium atom, uB is the Bohr magnetic, m is the mass of lithium
divided by 1kg, and uB' is the new Bohr magneton. uB' ~ 796.0
"""
_inch_To_Meter = lambda x: .0254 * x
gauss_To_Tesla = lambda x: x / 10_000.0

MASS_LITHIUM_7: float = 1.165034676538e-26  # mass of lithium 7, kg
MASS_HELIUM: float = 6.64216e-27  # mass of lithium 7, kg
BHOR_MAGNETON: float = 9.274009994e-24  # Bohr magneton, SI
BOLTZMANN_CONSTANT: float = 1.38064852e-23  # Boltzman constant, SI
MAGNETIC_PERMEABILITY: float = 1.25663706212 * 1e-6  # magnetic permeability, SI
GRAVITATIONAL_ACCELERATION: float = 9.8  # graviational acceleration on earth, nominal, SI
SIMULATION_MAGNETON: float = BHOR_MAGNETON / MASS_LITHIUM_7  # Simulation units bohr magneton. Described in model docstring
VACUUM_TUBE_THICKNESS: float = _inch_To_Meter(.05)  # minimum weldable thickness of a vacuum tube, m
COMBINER_VACUUM_TUBE_THICKNESS: float = 2*VACUUM_TUBE_THICKNESS  # minimum weldable thickness of a vacuum tube, m
DEFAULT_ATOM_SPEED: float = 210.0  # default speed of atoms, m/s
MIN_MAGNET_MOUNT_THICKNESS: float = 1e-3  # Nominal minimum material thickness for magnet mount, m. At thinnest point
FLAT_WALL_VACUUM_THICKNESS: float = _inch_To_Meter(1 / 8)  # minimum thickness of a flat wall under vacuum with
# dimension less  than 1x1 foot

SPACE_BETWEEN_MAGNETS_IN_MOUNT = .25e-3  # space between inside edges of neighboring magnets in hexapole mount
# configuration. Smaller than this and assembly could be infeasible because of magnet/machining tolerances

MAGNET_WIRE_DIAM: float = 500e-6  # this thickness supports ~.5amp/mm^2 (no water cooling) for 5 gauss in a solenoid
SPIN_FLIP_AVOIDANCE_FIELD = gauss_To_Tesla(3)  # field to prevent majorana splin flips

CUBIC_METER_TO_CUBIC_INCH = 61023.7  # conversion from cubic meters to cubic inches
COST_PER_CUBIC_INCH_PERM_MAGNET: float = 34.76  # USD. Per K&J magnetics for N52:
# https://www.kjmagnetics.com/proddetail.asp?prod=BX0X0X0-N52&cat=168

GRADE_MAGNETIZATION = {'N52': 1.465 / MAGNETIC_PERMEABILITY, 'N50': 1.43 / MAGNETIC_PERMEABILITY,
                       'N48': 1.40 / MAGNETIC_PERMEABILITY, 'N45': 1.35 / MAGNETIC_PERMEABILITY,
                       'N42': 1.31 / MAGNETIC_PERMEABILITY, 'N40': 1.275 / MAGNETIC_PERMEABILITY,
                       'legacy': 1.018E6}

# from, https://www.kjmagnetics.com/specs.asp




