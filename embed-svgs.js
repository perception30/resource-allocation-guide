const fs = require('fs');

// Read the HTML file
let html = fs.readFileSync('index.html', 'utf8');

// SVG files to embed
const svgFiles = [
    'resource-allocation-overview.svg',
    'allocation-algorithms.svg',
    'buddy-system.svg',
    'bankers-algorithm.svg',
    'hierarchical-allocation.svg',
    'state-transitions.svg',
    'performance-comparison.svg',
    'practical-implementation.svg'
];

// Read and embed each SVG
svgFiles.forEach((svgFile, index) => {
    const svgContent = fs.readFileSync(svgFile, 'utf8');
    const svgId = svgFile.replace('.svg', '').replace(/-/g, '_');
    
    // Create a data URL from the SVG content
    const base64Svg = Buffer.from(svgContent).toString('base64');
    const dataUrl = `data:image/svg+xml;base64,${base64Svg}`;
    
    // Store as JavaScript variable
    if (index === 0) {
        html = html.replace('// Load SVG diagrams', `// Embedded SVG data
        const svgData = {};
        svgData['${svgFile}'] = '${dataUrl}';
        
        // Load SVG diagrams`);
    } else {
        html = html.replace('// Load SVG diagrams', `svgData['${svgFile}'] = '${dataUrl}';
        
        // Load SVG diagrams`);
    }
});

// Update the diagram loading code
html = html.replace(
    `        diagrams.forEach(diagram => {
            fetch(\`\${diagram.file}.base64\`)
                .then(response => response.text())
                .then(base64 => {
                    document.getElementById(diagram.id).src = \`data:image/svg+xml;base64,\${base64}\`;
                })
                .catch(err => console.error(\`Error loading \${diagram.file}:\`, err));
        });`,
    `        diagrams.forEach(diagram => {
            if (svgData[diagram.file]) {
                document.getElementById(diagram.id).src = svgData[diagram.file];
            } else {
                console.error(\`SVG not found: \${diagram.file}\`);
            }
        });`
);

// Write the updated HTML
fs.writeFileSync('index-embedded.html', html);
console.log('Created index-embedded.html with embedded SVGs');