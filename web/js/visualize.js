$(document).ready(function() {
    var jsonData = {
        'total': {'Computer': 291821,
                  'Unknown': 5,
                  'Unspecified': 1899,
                  'Mobile': 390598,
                  'Tablet': 222191,
                  'Big screen': 29},
        'sso': {'Unspecified': 228,
                'Mobile': 46439,
                'Tablet': 39986,
                'Computer': 54292},
        'primary': {'Unspecified': 26,
                    'Mobile': 4171,
                    'Tablet': 4782,
                    'Computer': 5099}
    };

    var totalPlatform = jsonData.total;

    // D3
    var svg = d3.select("#platform"),
        margin = {top: 20, right: 20, bottom: 30, left: 50},
        width = +svg.attr("width") - margin.left - margin.right,
        height = +svg.attr("height") - margin.top - margin.bottom;

    var x = d3.scaleBand().rangeRound([0, width]).padding(0.1),
        y = d3.scaleLinear().rangeRound([height, 0]);

    var g = svg.append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
    
    x.domain(Object.keys(totalPlatform));
    y.domain([0, d3.max(Object.values(totalPlatform))]);

    g.append("g")
        .attr("class", "axis axis--x")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x));

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
        .data(Object.values(totalPlatform))
        .enter().append("rect")
        .attr("class", "bar")
        .attr("x", function(d, i) { return x(Object.keys(totalPlatform)[i]); })
        .attr("y", function(d, i) { return y(Object.values(totalPlatform)[i]); })
        .attr("width", x.bandwidth())
        .attr("height", function(d) { return height - y(d); });
})
