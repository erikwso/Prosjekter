import numpy as np
from scipy.stats import norm

def black_scholes_call_price(S, K, T, r, sigma):
    """Black-Scholes price for a European Call option."""
    S = np.asarray(S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

# --- Default Parameters ---
S0 = 100     # Initial spot
K = 105      # Strike
T = 1.0      # Time to maturity (years)
r = 0.04     # Risk-free rate
sigma = 0.24 # Volatility

# --- Generate HTML file with interactive controls ---
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Black-Scholes Monte Carlo Pricing</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #0a1929;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #ffffff;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #ffffff;
            margin-bottom: 30px;
        }}
        .controls {{
            background: #1e3a5f;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        .input-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .input-group {{
            display: flex;
            flex-direction: column;
        }}
        label {{
            margin-bottom: 8px;
            font-size: 14px;
            color: #ffffff;
            font-weight: 500;
        }}
        input {{
            padding: 10px;
            border: 2px solid #4a9eff;
            border-radius: 5px;
            background: #0a1929;
            color: #ffffff;
            font-size: 16px;
            transition: border-color 0.3s;
        }}
        input:focus {{
            outline: none;
            border-color: #6bb6ff;
        }}
        .button-container {{
            text-align: center;
        }}
        button {{
            background: #4a9eff;
            color: white;
            border: none;
            padding: 12px 40px;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(74, 158, 255, 0.4);
            background: #6bb6ff;
        }}
        button:active {{
            transform: translateY(0);
        }}
        #plot {{
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Black-Scholes Monte Carlo Option Pricing</h1>
        
        <div class="controls">
            <div class="input-grid">
                <div class="input-group">
                    <label for="S0">Initial Spot (S0)</label>
                    <input type="number" id="S0" value="{S0}" step="1" min="0">
                </div>
                <div class="input-group">
                    <label for="K">Strike Price (K)</label>
                    <input type="number" id="K" value="{K}" step="1" min="0">
                </div>
                <div class="input-group">
                    <label for="T">Time to Maturity (years)</label>
                    <input type="number" id="T" value="{T}" step="0.1" min="0.01">
                </div>
                <div class="input-group">
                    <label for="r">Risk-Free Rate</label>
                    <input type="number" id="r" value="{r}" step="0.01" min="0">
                </div>
                <div class="input-group">
                    <label for="sigma">Volatility (sigma)</label>
                    <input type="number" id="sigma" value="{sigma}" step="0.01" min="0.01">
                </div>
            </div>
            <div class="button-container">
                <button onclick="runSimulation()">Calculate</button>
            </div>
        </div>
        
        <div id="plot"></div>
    </div>

    <script>
        // Normal CDF approximation
        function normCDF(x) {{
            const t = 1 / (1 + 0.2316419 * Math.abs(x));
            const d = 0.3989423 * Math.exp(-x * x / 2);
            const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
            return x > 0 ? 1 - p : p;
        }}

        // Black-Scholes call price
        function blackScholesCall(S, K, T, r, sigma) {{
            const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
            const d2 = d1 - sigma * Math.sqrt(T);
            return S * normCDF(d1) - K * Math.exp(-r * T) * normCDF(d2);
        }}

        // Box-Muller transform for normal random variables
        function randn() {{
            const u1 = Math.random();
            const u2 = Math.random();
            return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        }}

        function runSimulation() {{
            // Get parameters
            const S0 = parseFloat(document.getElementById('S0').value);
            const K = parseFloat(document.getElementById('K').value);
            const T = parseFloat(document.getElementById('T').value);
            const r = parseFloat(document.getElementById('r').value);
            const sigma = parseFloat(document.getElementById('sigma').value);

            // Black-Scholes curve
            const S_range = [];
            const bs_values = [];
            for (let S = 60; S <= 140; S += 0.8) {{
                S_range.push(S);
                bs_values.push(blackScholesCall(S, K, T, r, sigma));
            }}
            const bs_price = blackScholesCall(S0, K, T, r, sigma);

            // Monte Carlo simulation
            const N_mc = 200000;
            let payoff_sum = 0;
            let payoff_sq_sum = 0;

            for (let i = 0; i < N_mc; i++) {{
                const Z = randn();
                const ST = S0 * Math.exp((r - 0.5 * sigma * sigma) * T + sigma * Math.sqrt(T) * Z);
                const payoff = Math.max(ST - K, 0);
                payoff_sum += payoff;
                payoff_sq_sum += payoff * payoff;
            }}

            const mc_price = Math.exp(-r * T) * (payoff_sum / N_mc);
            const payoff_var = (payoff_sq_sum / N_mc) - Math.pow(payoff_sum / N_mc, 2);
            const mc_se = Math.exp(-r * T) * Math.sqrt(payoff_var / N_mc);
            const ci_low = mc_price - 1.96 * mc_se;
            const ci_high = mc_price + 1.96 * mc_se;

            // Generate sample paths
            const N_paths = 8;
            const N_steps = 120;
            const dt = T / N_steps;
            const S_paths = [];
            const t_grid = [];

            for (let i = 0; i <= N_steps; i++) {{
                t_grid.push(i * dt);
            }}

            for (let path = 0; path < N_paths; path++) {{
                const S_path = [S0];
                let W = 0;
                
                for (let step = 1; step <= N_steps; step++) {{
                    const dW = randn() * Math.sqrt(dt);
                    W += dW;
                    const X = (r - 0.5 * sigma * sigma) * t_grid[step] + sigma * W;
                    S_path.push(S0 * Math.exp(X));
                }}
                S_paths.push(S_path);
            }}

            // Create plotly figure
            const traces = [];

            // Left panel: BS curve
            traces.push({{
                x: S_range,
                y: bs_values,
                mode: 'lines',
                line: {{ color: '#4a9eff', width: 4 }},
                name: 'BS Value',
                xaxis: 'x',
                yaxis: 'y'
            }});

            // S0 vertical line
            traces.push({{
                x: [S0, S0],
                y: [0, bs_price],
                mode: 'lines',
                line: {{ color: '#6bb6ff', dash: 'dot', width: 3 }},
                showlegend: false,
                xaxis: 'x',
                yaxis: 'y'
            }});

            // BS point
            traces.push({{
                x: [S0],
                y: [bs_price],
                mode: 'markers+text',
                marker: {{ size: 10, color: '#6bb6ff' }},
                text: [`BS: ${{bs_price.toFixed(2)}}`],
                textposition: 'top right',
                textfont: {{ size: 14, color: '#ffffff' }},
                showlegend: false,
                xaxis: 'x',
                yaxis: 'y'
            }});

            // MC point
            traces.push({{
                x: [S0],
                y: [mc_price],
                mode: 'markers+text',
                marker: {{ size: 10, color: '#4a9eff' }},
                text: [`MC: ${{mc_price.toFixed(2)}}`],
                textposition: 'bottom right',
                textfont: {{ size: 14, color: '#ffffff' }},
                name: 'MC Price',
                xaxis: 'x',
                yaxis: 'y'
            }});

            // Right panel: Sample paths
            for (let i = 0; i < N_paths; i++) {{
                traces.push({{
                    x: t_grid,
                    y: S_paths[i],
                    mode: 'lines',
                    line: {{ width: 2, color: '#6bb6ff' }},
                    opacity: 0.6,
                    showlegend: false,
                    xaxis: 'x2',
                    yaxis: 'y2'
                }});
            }}

            // Strike line
            traces.push({{
                x: [0, T],
                y: [K, K],
                mode: 'lines',
                line: {{ color: '#ffffff', width: 3, dash: 'dash' }},
                name: 'Strike K',
                xaxis: 'x2',
                yaxis: 'y2'
            }});

            // Maturity line
            const all_S_values = S_paths.flat();
            const min_S = Math.min(...all_S_values, S0 * 0.4);
            const max_S = Math.max(...all_S_values, S0 * 1.6);
            traces.push({{
                x: [T, T],
                y: [min_S, max_S],
                mode: 'lines',
                line: {{ color: '#4a9eff', width: 3, dash: 'dot' }},
                name: 'Maturity T',
                xaxis: 'x2',
                yaxis: 'y2'
            }});

            const layout = {{
                height: 500,
                width: 1150,
                plot_bgcolor: '#1e3a5f',
                paper_bgcolor: '#0a1929',
                font: {{ color: '#ffffff', size: 16 }},
                title: {{
                    text: `<br>Black-Scholes Value and Monte Carlo Simulations<br>` +
                          `<span style='font-size:18px;color:#6bb6ff'>` +
                          `BS(S0)=${{bs_price.toFixed(4)}} | MC=${{mc_price.toFixed(4)}} ` +
                          `(95% CI: [${{ci_low.toFixed(4)}}, ${{ci_high.toFixed(4)}}], N=${{N_mc.toLocaleString()}})` +
                          `</span><br>`,
                    y: 0.98,
                    x: 0.5,
                    font: {{ color: '#ffffff', size: 25 }}
                }},
                margin: {{ l: 44, r: 26, t: 95, b: 95 }},
                legend: {{
                    orientation: 'h',
                    yanchor: 'top',
                    y: -0.30,
                    xanchor: 'center',
                    x: 0.5,
                    font: {{ size: 16 }}
                }},
                grid: {{
                    rows: 1,
                    columns: 2,
                    pattern: 'independent',
                    roworder: 'top to bottom'
                }},
                xaxis: {{
                    title: 'Spot Price S0',
                    domain: [0, 0.48],
                    range: [58, 142],
                    showgrid: true,
                    gridcolor: 'rgba(255,255,255,0.1)'
                }},
                yaxis: {{
                    title: 'Option Value C(S0, 0)',
                    showgrid: true,
                    gridcolor: 'rgba(255,255,255,0.1)'
                }},
                xaxis2: {{
                    title: 'Time t',
                    domain: [0.55, 1],
                    range: [-0.01, T + 0.02],
                    showgrid: true,
                    gridcolor: 'rgba(255,255,255,0.1)'
                }},
                yaxis2: {{
                    title: 'Spot Price St',
                    anchor: 'x2',
                    range: [Math.min(50, min_S - 5), Math.max(150, max_S + 5)],
                    showgrid: true,
                    gridcolor: 'rgba(255,255,255,0.1)'
                }}
            }};

            Plotly.newPlot('plot', traces, layout);
        }}

        // Run on page load
        runSimulation();
    </script>
</body>
</html>
"""

# Write HTML file
with open('option_pricing_interactive.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Interactive HTML file created: option_pricing_interactive.html")
print("Open this file in your web browser to use the interactive interface!")