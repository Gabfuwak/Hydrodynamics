# SPH implementation report
**Author:** Gabin Maury
## Summary

A SPH simulation simulates fluid by modeling it as a set of interacting particles.

The particles have different forces applied to them: gravity, pressure, viscosity. Since the fluid we are simulating is incompressible, the main force applied comes from pressure, preventing the particles from forming a compressed fluid. 
## Implementation Details

### Force Application

Different forces are applied to the particles.

**Body Force**
Body force currently only includes gravity. This would also include the force applied by objects in the water in a more advanced simulation

**Pressure Force**
Pressure force is applied by the particles "pushing" on each other. This is the main force that gives the simulation its "fluid" look. The particles get a force proportional to the inverse of the gradient of pressure.

The pressure value depends on the density passed through an equation of state. This equation depends on the fluid we are trying to simulate.

**Viscosity Force**
Viscosity is how much the particles "stick" to one an other. This is represented as a small force from the particle towards its neighbors. A high viscosity would make our fluid look like honey.

To improve stability and prevent division by zero or very small numbers when particles are very close, a safety check was added for pressure and viscosity:

```cpp
if (len < 1e-10) {
    continue;  // Skip extremely close particles
}
float density_i = std::max(_density[i], 1e-6f);
float density_j = std::max(_density[neigh_particle], 1e-6f);
```

This improves the stability of the simulation slightly, especially for high viscosity fluids.
### Boundary Handling

The simulation uses "hard" boundaries that particles cannot traverse. To make it more physically accurate, we add immobile particles in the boundary. They are there to make pressure calculation more accurate. In a more advanced simulation we could calculate pressure using an integral on the smoothing kernel.

To optimize performance, boundary particles are stored in the first `_numBoundaryParticles` elements of the particle arrays. This allows us to iterate from `_numBoundaryParticles` to `_pos.size()` instead of `0` to `_pos.size()`, eliminating the need for a conditional check within the loop and slightly reducing memory usage.

```cpp
for(int i=_numBoundaryParticles; i<particleCount(); ++i) {
    // ... collision checks and handling ...
}
```

## Visual improvements

The original code had a color modification depending on density, changing the blue level. This had an issue where low density turned the particles yellow because we removed all the blue channel.

Instead of working on the blue channel, i decided to set it permanently to 1, and make the other vary. The lower the density, the higher the green and red channel, making the particle light blue when the density is low, and deep blue when the density is high.

In addition, I made the particles bigger, matching their size in the simulation.

These changes makes the simulation much more aesthetically pleasing, with low density white particles emulating a sort of sea foam, which pairs well with a little higher viscosity, creating beautiful waves. 
![image](https://github.com/user-attachments/assets/e93d346f-ff15-45b1-b504-1e06ae1d45cb)

## Results

### Simulation Stability

While stability is still an issue and the water fluid to feel like it's "boiling" due to the lack of surface tension, the simulation remains stable under most conditions (breaks at ~300 of viscosity) and always end up settling down. The inclusion of safety checks in pressure and viscosity calculations prevents numerical issues arising from near-zero distances between particles.

### Performance

The use of a grid-based neighbor search and the optimization of storing boundary particles at the beginning of arrays contribute to the overall efficiency of the simulation. After the addition of openMP paralization, the simulation is easily able to run faster than real time - real time being 200 steps per second - on my computer (13th gen i7 CPU) with around 4000 particle.

### Visual Output

The simulation produces a visually plausible fluid behavior. Particles move and interact in a way that is consistent with the applied forces and constraints. The use of a color gradient to represent density provides a clear visualization of density variations within the fluid, even mimicking sea foam in some cases.


## Perspectives and limitations

In my opinion, the main issue of this simulation is the lack of surface tension. It makes the fluid lack coherence and feel like it's "boiling" when the viscosity is low.

I tried to fix this issue by creating a signed distance field, applying the level set technique to find the boundary of the fluid and make surface tension possible. Sadly i was not able to complete the implementation in time.

An other possibility of the signed distance field and fluid boundary detection is to greatly improve the visual output with a smooth liquid rather than particles.


