import pandapower as pp

net = pp.create_empty_network(sn_mva=100)

buses = {}
for name in ["NORTE", "NORDESTE", "SUDESTE", "SUL", "PARAGUAI"]:
    buses[name] = pp.create_bus(net, name=f"BUS_{name}", vn_kv=230.0)

# tie at SUDESTE acts as reference
pp.create_ext_grid(net, bus=buses["SUDESTE"], vm_pu=1.0, name="GRID_SUDESTE")

# define simplified interconnections (length_km, r, x, max_i)
connections = [
    ("NORTE", "NORDESTE", 400, 0.04, 0.4, 1.5),
    ("NORTE", "SUDESTE", 800, 0.04, 0.4, 1.2),
    ("NORDESTE", "SUDESTE", 500, 0.03, 0.3, 1.4),
    ("NORDESTE", "SUL", 900, 0.05, 0.45, 1.0),
    ("SUDESTE", "SUL", 350, 0.02, 0.2, 1.6),
    ("PARAGUAI", "SUDESTE", 100, 0.01, 0.1, 1.8),
    ("PARAGUAI", "SUL", 250, 0.02, 0.2, 1.0)
]

for fr, to, length, r, x, i_max in connections:
    pp.create_line_from_parameters(
        net,
        from_bus=buses[fr],
        to_bus=buses[to],
        length_km=length,
        r_ohm_per_km=r,
        x_ohm_per_km=x,
        c_nf_per_km=0.0,
        max_i_ka=i_max,
        name=f"LINE_{fr}_{to}",
        type="ol"
    )

pp.to_json(net, "models/brazil_5bus_network.json")
print("Saved simplified 5-bus network to models/brazil_5bus_network.json")
