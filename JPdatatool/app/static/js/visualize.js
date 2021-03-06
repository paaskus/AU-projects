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

function sortByValue(object) {
    // convert object into array
    var sorted = [];
    for (var key in object)
	sorted.push([key, object[key]]); // each item is an array in format [key, value]

    console.log(sorted);
    
    // sort items by value
    sorted.sort(function(a, b) {
	return Math.max(b[1].normal, b[1].mapped) - Math.max(a[1].normal, a[1].mapped); // compare numbers
    });
    
    return sorted; // array in format [ [ key1, val1 ], [ key2, val2 ], ... ]
}

// D3
function renderBarChart(containerID, barchartdata, chartname) {
    var sortedData = sortByValue(barchartdata);
    var keys = sortedData.map(function(obj) {
	return obj[0] // keys
    });
    var values = sortedData.map(function(obj) {
	return obj[1].normal // values
    });
    var mappedValues = sortedData.map(function(obj) {
	return obj[1].mapped // values
    });

    values = mappedValues.concat(values);

//    console.log("value length: "+values.length);
//    console.log("mapping length: "+mappedValues.length);
    
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

    // insert mapped data
    g.selectAll(".bar")
        .data(values)
        .enter().append("rect")
        .attr("class", "bar")
        .attr("x", function(d, i) { return x(keys[i % keys.length]); })
        .attr("y", function(d, i) { return y(values[i]); })
        .attr("width", x.bandwidth())
        .attr("height", function(d) { return height - y(d); })
        .style("fill", function(d, i) { return (i < keys.length) ? 'rgba(0, 0, 0, 1)' : 'rgba(12, 255, 158, 0.7)' ; });

    g.append("text")
        .attr("x", (width / 2))             
        .attr("y", 0 - (margin.top / 2))
        .attr("text-anchor", "middle")  
        .style("font-size", "16px") 
        .text(chartname);
};
