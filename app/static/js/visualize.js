function drawCharts(crunchedData) {
    var data = crunchedData;
    
    var platformData = data['Platform'];
    render3BarCharts(platformData, 'Platform');

    var deviceData = data['Device'];
    render3BarCharts(deviceData, 'Device');

    var operatingSystemData = data['Operating system'];
    render3BarCharts(operatingSystemData, 'Operating system');

    var webBrowserData = data['Web browser'];
    render3BarCharts(webBrowserData, 'Web browser');

    var categoryData = data['Category'];
    render3BarCharts(categoryData, 'Category');
};

function render3BarCharts(data, attributeName) {
    var containerID = attributeNameToID(attributeName);
    renderBarChart(containerID, data.total, 'Total');
    renderBarChart(containerID, data.sso, 'SSO only');
    renderBarChart(containerID, data.primary, 'Primary ' + attributeName.toLowerCase());
}

/* 
 * transforms a string of form 'Operating system' to a string of form '#operating-system'
 */
function attributeNameToID(attributeName) {
    var lowercase = attributeName.toLowerCase();
    var hyphenated = lowercase.split(' ').join('-');
    return '#'+hyphenated;
}

// D3
function renderBarChart(containerID, barchartdata, chartname) {
    var keys = Object.keys(barchartdata);
    var values = Object.values(barchartdata);
    
    var $svg = $('<svg class="chart" width="380" height="450" viewBox="0 0 380 450" preerveAspectRatio="xMinYMin meet"></svg>');
    $(containerID).append($svg);
    var svg = d3.select($svg[0]),
        margin = {top: 40, right: 20, bottom: 150, left: 50},
        width = +svg.attr("width") - margin.left - margin.right,
        height = +svg.attr("height") - margin.top - margin.bottom;

    var x = d3.scaleBand().rangeRound([0, width]).padding(0.1),
        y = d3.scaleLinear().rangeRound([height, 0]);

    var g = svg.append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
    
    x.domain(keys);
    y.domain([0, d3.max(values)]);

    g.append("g")
        .attr("class", "axis axis--x")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x))
        .selectAll("text")
        .attr("y", 0)
        .attr("x", 9)
        .attr("dy", ".35em")
        .attr("transform", "rotate(90)")
        .style("text-anchor", "start");

    g.append("g")
        .attr("class", "axis axis--y")
        .call(d3.axisLeft(y).ticks(10, "").tickFormat(d3.format(",d")))
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 6)
        .attr("dy", "0.71em")
        .attr("text-anchor", "end")
        .text("Frequency");

    g.selectAll(".bar")
        .data(values)
        .enter().append("rect")
        .attr("class", "bar")
        .attr("x", function(d, i) { return x(keys[i]); })
        .attr("y", function(d, i) { return y(values[i]); })
        .attr("width", x.bandwidth())
        .attr("height", function(d) { return height - y(d); });

    g.append("text")
        .attr("x", (width / 2))             
        .attr("y", 0 - (margin.top / 2))
        .attr("text-anchor", "middle")  
        .style("font-size", "16px") 
        .text(chartname);
};
