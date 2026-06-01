# Design System Guidelines: Photocatalyst HER

## Theme Definition

```yaml
name: Photocatalyst HER
colors:
  surface: '#051424'
  surface-dim: '#051424'
  surface-bright: '#2c3a4c'
  surface-container-lowest: '#010f1f'
  surface-container-low: '#0d1c2d'
  surface-container: '#122131'
  surface-container-high: '#1c2b3c'
  surface-container-highest: '#273647'
  on-surface: '#d4e4fa'
  on-surface-variant: '#bdc8d1'
  inverse-surface: '#d4e4fa'
  inverse-on-surface: '#233143'
  outline: '#87929a'
  outline-variant: '#3e484f'
  surface-tint: '#7bd0ff'
  primary: '#8ed5ff'
  on-primary: '#00354a'
  primary-container: '#38bdf8'
  on-primary-container: '#004965'
  inverse-primary: '#00668a'
  secondary: '#4edea3'
  on-secondary: '#003824'
  secondary-container: '#00a572'
  on-secondary-container: '#00311f'
  tertiary: '#e1bfff'
  on-tertiary: '#490080'
  tertiary-container: '#ce9bff'
  on-tertiary-container: '#6400ac'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#c4e7ff'
  primary-fixed-dim: '#7bd0ff'
  on-primary-fixed: '#001e2c'
  on-primary-fixed-variant: '#004c69'
  secondary-fixed: '#6ffbbe'
  secondary-fixed-dim: '#4edea3'
  on-secondary-fixed: '#002113'
  on-secondary-fixed-variant: '#005236'
  tertiary-fixed: '#f0dbff'
  tertiary-fixed-dim: '#ddb7ff'
  on-tertiary-fixed: '#2c0051'
  on-tertiary-fixed-variant: '#6900b3'
  background: '#051424'
  on-background: '#d4e4fa'
  surface-variant: '#273647'
typography:
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  data-lg:
    fontFamily: JetBrains Mono
    fontSize: 18px
    fontWeight: '500'
    lineHeight: 24px
  data-md:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  base: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  container-max: 1440px
  gutter: 16px
```

---

## Brand & Style

The design system is engineered for a high-performance scientific environment focused on Hydrogen Evolution Reaction (HER) research. The brand personality is **technical, precise, and authoritative**, designed to minimize cognitive load during complex data analysis while maintaining a sophisticated aesthetic.

The visual style is **Obsidian-Slate Glassmorphism**. This approach utilizes deep, layered backgrounds to provide a sense of infinite depth, punctuated by vibrant, high-contrast functional accents. The UI feels like a high-end laboratory instrument—utilitarian but refined. Key characteristics include:
- **Optical Depth:** Use of semi-transparent layers to stack information without clutter.
- **Scientific Precision:** Sharp contrasts and technical typefaces to ensure data legibility.
- **Responsive Feedback:** Micro-interactions that provide tactile confirmation of data changes.

## Colors

The palette is rooted in an **Obsidian-Slate** dark mode to reduce eye strain during long-duration research sessions.

- **Primary (Ice Blue):** Used for primary actions, active states, and core metric highlights. It signifies interactive potential.
- **Secondary (Emerald Green):** Reserved strictly for "Success" states, targets met, and positive catalyst performance indicators.
- **Tertiary (Neon Violet):** Specifically designated for feature importance, ML model insights, and experimental variables.
- **Neutral (Slate):** A range of greys used for borders, secondary text, and inactive states to maintain hierarchy.
- **Backgrounds:** The base layer (#0B0E14) provides the foundation, while the surface layer (#151921) defines the cards and containers.

## Typography

This design system employs a dual-font strategy. **Inter** handles all UI labels, navigation, and instructional text for maximum readability. **JetBrains Mono** is utilized for all scientific data, ML parameters, and coordinate values to ensure characters remain distinct and vertically aligned in data grids.

- **Headlines:** Bold and tight for clear section definitions.
- **Data Roles:** Use `data-md` for standard grid entries and `data-lg` for primary hero metrics.
- **Labels:** The `label-caps` role is intended for table headers and small metadata tags.

## Layout & Spacing

The layout follows a **structured 12-column fluid grid** for desktop, optimized for data-dense dashboards. 

- **Dashboard Layout:** Utilizes a fixed left sidebar (240px) for navigation and a fluid main content area.
- **Spacing Rhythm:** Based on a 4px baseline. Components should use `md` (16px) for internal padding and `lg` (24px) for external margins to maintain a clean, airy feel despite the high density of information.
- **Breakpoints:** 
    - Desktop: 12 columns, 24px margins.
    - Tablet: 8 columns, 16px margins.
    - Mobile: 4 columns, 16px margins; metrics cards stack vertically.

## Elevation & Depth

Hierarchy is established through **Tonal Layering** and **Glassmorphism**. Shadows are avoided in favor of light-based depth markers.

- **Level 0 (Base):** Deepest charcoal (#0B0E14). Used for the global canvas.
- **Level 1 (Surfaces):** Slate surface (#151921). Used for primary cards and content containers.
- **Level 2 (Overlays/Modals):** A semi-transparent blur (Backdrop-filter: blur(12px)) with a subtle 1px border (#FFFFFF10). This creates the "Glass" effect for floating panels and filters.
- **Interaction Depth:** When hovered, elements should increase their border brightness rather than their shadow spread to maintain a "technical" feel.

## Shapes

The design system uses a **Soft** shape language to balance the "cold" technical nature of the data with modern UI approachability.

- **Standard Components:** 0.25rem (4px) corner radius for buttons and input fields to maintain a crisp, precise look.
- **Containers/Cards:** 0.5rem (8px) corner radius for metrics cards and data grids.
- **Interactive Chips:** Fully rounded (pill-shaped) to distinguish them from structural data elements.

## Components

### Micro-cards (Metrics)
Used for displaying high-level KPIs like "H2 Yield" or "Efficiency %." 
- **Style:** Background #151921 with a 1px top-border highlight in Ice Blue or Emerald Green.
- **Typography:** Large `data-lg` for the value, `label-caps` for the description.

### Reactive Data Grids
- **Header:** Sticky headers with a semi-transparent slate background and `label-caps` typography.
- **Rows:** Subtle 1px bottom border (#FFFFFF05). Row hover state uses a subtle primary color tint (5% opacity).
- **Cells:** Numeric data must use `jetbrainsMono`.

### Advanced Filter Chips
- **Inactive:** Transparent background with a Slate border.
- **Active:** Ice Blue background with #0B0E14 text.
- **Shape:** Pill-shaped for easy identification.

### Scientific Charts
- **Grid Lines:** Low-contrast Slate (#FFFFFF10).
- **Primary Data Path:** 2px stroke width in Ice Blue.
- **Confidence Intervals:** 10% opacity fill of the primary color.

### Input Fields
- **Style:** Filled style with #151921 background. Bottom-only border (2px) that glows Ice Blue on focus.
- **Placeholder:** Low-contrast Slate (#475569).
