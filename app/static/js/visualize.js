function drawCharts(crunchedData) {
    var data = crunchedData;
    
    var platformData = data['Platform'];

    renderBarChart('#platform', platformData.total, 'Total');
    renderBarChart('#platform', platformData.sso, 'SSO only');
    renderBarChart('#platform', platformData.primary, 'Primary platform');

    var deviceData = data['Device'];
    
    renderBarChart('#device', deviceData.total, 'Total');
    renderBarChart('#device', deviceData.sso, 'SSO only');
    renderBarChart('#device', deviceData.primary, 'Primary device');

    var operatingSystemData = data['Operating system'];

    renderBarChart('#operating-system', operatingSystemData.total, 'Total');
    renderBarChart('#operating-system', operatingSystemData.sso, 'SSO only');
    renderBarChart('#operating-system', operatingSystemData.primary, 'Primary operating system');

    var webBrowserData = data['Web browser'];

    renderBarChart('#web-browser', webBrowserData.total, 'Total');
    renderBarChart('#web-browser', webBrowserData.sso, 'SSO only');
    renderBarChart('#web-browser', webBrowserData.primary, 'Primary web browser');

    var categoryData = data['Category'];

    renderBarChart('#category', categoryData.total, 'Total');
    renderBarChart('#category', categoryData.sso, 'SSO only');
    renderBarChart('#category', categoryData.primary, 'Primary category');
};

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
