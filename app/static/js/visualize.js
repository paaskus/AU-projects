function drawCharts(crunchedData) {
    var data = crunchedData;
    data = {"Device": {"total": {"Java Mobile": {"mapped": 63, "normal": 63}, "Kindle": {"mapped": 78, "normal": 78}, "Windows Desktop": {"mapped": 231049, "normal": 231049}, "Macintosh Desktop": {"mapped": 55366, "normal": 55366}, "Chromium OS Desktop": {"mapped": 1131, "normal": 1131}, "iPod": {"mapped": 91, "normal": 91}, "Android Device": {"mapped": 10139, "normal": 10139}, "Android Mobile": {"mapped": 143018, "normal": 143018}, "Android Tablet": {"mapped": 23360, "normal": 23360}, "Android Laptop": {"mapped": 1, "normal": 1}, "FreeBSD Desktop": {"mapped": 3, "normal": 3}, "Symbian Mobile": {"mapped": 138, "normal": 138}, "Android Desktop": {"mapped": 1, "normal": 1}, "iPhone": {"mapped": 227243, "normal": 227243}, "Console": {"mapped": 11, "normal": 11}, "Linux Desktop": {"mapped": 4275, "normal": 4275}, "Unspecified": {"mapped": 1, "normal": 1}, "Windows Mobile": {"mapped": 12405, "normal": 12405}, "Windows Tablet": {"mapped": 1623, "normal": 1623}, "BlackBerry": {"mapped": 82, "normal": 82}, "iPad": {"mapped": 196260, "normal": 196260}, "Unknown": {"mapped": 189, "normal": 189}, "TV": {"mapped": 16, "normal": 16}}, "sso": {"Symbian Mobile": {"mapped": 14, "normal": 12}, "BlackBerry": {"mapped": 21, "normal": 19}, "Java Mobile": {"mapped": 51, "normal": 0}, "iPad": {"mapped": 55866, "normal": 34596}, "iPhone": {"mapped": 48484, "normal": 23831}, "Windows Desktop": {"mapped": 49482, "normal": 43212}, "Macintosh Desktop": {"mapped": 12197, "normal": 10250}, "Chromium OS Desktop": {"mapped": 302, "normal": 279}, "Linux Desktop": {"mapped": 615, "normal": 551}, "Unspecified": {"mapped": 1, "normal": 0}, "iPod": {"mapped": 23, "normal": 7}, "Kindle": {"mapped": 34, "normal": 30}, "Windows Tablet": {"mapped": 361, "normal": 312}, "Android Device": {"mapped": 540, "normal": 449}, "Android Mobile": {"mapped": 23184, "normal": 20588}, "Android Tablet": {"mapped": 5631, "normal": 4969}, "Unknown": {"mapped": 22, "normal": 15}, "Windows Mobile": {"mapped": 2048, "normal": 1825}}, "primary": {"Symbian Mobile": {"mapped": 2, "normal": 2}, "BlackBerry": {"mapped": 1, "normal": 1}, "iPad": {"mapped": 3391, "normal": 3492}, "iPhone": {"mapped": 3187, "normal": 3247}, "Windows Desktop": {"mapped": 4501, "normal": 4525}, "Macintosh Desktop": {"mapped": 1076, "normal": 1081}, "Chromium OS Desktop": {"mapped": 25, "normal": 25}, "Linux Desktop": {"mapped": 35, "normal": 36}, "iPod": {"mapped": 2, "normal": 2}, "Kindle": {"mapped": 2, "normal": 2}, "Windows Tablet": {"mapped": 35, "normal": 39}, "Android Device": {"mapped": 20, "normal": 20}, "Android Mobile": {"mapped": 1035, "normal": 1039}, "Android Tablet": {"mapped": 377, "normal": 380}, "Unknown": {"mapped": 2, "normal": 2}, "Windows Mobile": {"mapped": 187, "normal": 185}}}, "Category": {"total": {"guide": {"mapped": 571, "normal": 571}, "jptv": {"mapped": 224, "normal": 224}, "anmeldelser": {"mapped": 9417, "normal": 9417}, "kommentarer": {"mapped": 3117, "normal": 3117}, "briefing": {"mapped": 245, "normal": 245}, "politik": {"mapped": 18359, "normal": 18359}, "section": {"mapped": 159, "normal": 159}, "topic": {"mapped": 548, "normal": 548}, "debat": {"mapped": 53329, "normal": 53329}, "mitjp": {"mapped": 1511, "normal": 1511}, "livsstil": {"mapped": 48259, "normal": 48259}, "frontpage": {"mapped": 152, "normal": 152}, "kultur": {"mapped": 16127, "normal": 16127}, "erhverv": {"mapped": 4867, "normal": 4867}, "advertorial": {"mapped": 206, "normal": 206}, "sport": {"mapped": 50082, "normal": 50082}, "download": {"mapped": 203, "normal": 203}, "indland": {"mapped": 62775, "normal": 62775}, "abnjp": {"mapped": 1965, "normal": 1965}, "foto": {"mapped": 88, "normal": 88}, "aarhus": {"mapped": 55898, "normal": 55898}, "indblik": {"mapped": 16798, "normal": 16798}, "navne": {"mapped": 1338, "normal": 1338}, "article": {"mapped": 2013, "normal": 2013}, "faktameter": {"mapped": 1, "normal": 1}, "international": {"mapped": 73818, "normal": 73818}, "timeout": {"mapped": 1314, "normal": 1314}, "viden": {"mapped": 9142, "normal": 9142}, "feature": {"mapped": 1, "normal": 1}, "other": {"mapped": 474016, "normal": 474016}}, "sso": {"guide": {"mapped": 212, "normal": 192}, "jptv": {"mapped": 12, "normal": 6}, "anmeldelser": {"mapped": 2741, "normal": 2619}, "kommentarer": {"mapped": 1252, "normal": 1141}, "briefing": {"mapped": 71, "normal": 68}, "politik": {"mapped": 3322, "normal": 2005}, "section": {"mapped": 90, "normal": 47}, "topic": {"mapped": 125, "normal": 84}, "debat": {"mapped": 8772, "normal": 5931}, "mitjp": {"mapped": 1511, "normal": 1511}, "livsstil": {"mapped": 9571, "normal": 6686}, "frontpage": {"mapped": 122, "normal": 31}, "kultur": {"mapped": 3205, "normal": 2459}, "erhverv": {"mapped": 1849, "normal": 1780}, "advertorial": {"mapped": 43, "normal": 19}, "sport": {"mapped": 11222, "normal": 8130}, "download": {"mapped": 203, "normal": 201}, "indland": {"mapped": 14091, "normal": 11779}, "abnjp": {"mapped": 1965, "normal": 1965}, "foto": {"mapped": 13, "normal": 10}, "aarhus": {"mapped": 9402, "normal": 5927}, "indblik": {"mapped": 5213, "normal": 4929}, "navne": {"mapped": 419, "normal": 401}, "article": {"mapped": 657, "normal": 607}, "international": {"mapped": 15256, "normal": 11263}, "timeout": {"mapped": 206, "normal": 148}, "viden": {"mapped": 2108, "normal": 1706}, "other": {"mapped": 105223, "normal": 69300}}, "primary": {"guide": {"mapped": 7, "normal": 7}, "abnjp": {"mapped": 291, "normal": 195}, "anmeldelser": {"mapped": 29, "normal": 29}, "kommentarer": {"mapped": 35, "normal": 35}, "briefing": {"mapped": 10, "normal": 10}, "politik": {"mapped": 748, "normal": 755}, "section": {"mapped": 4, "normal": 7}, "debat": {"mapped": 94, "normal": 93}, "mitjp": {"mapped": 0, "normal": 140}, "livsstil": {"mapped": 220, "normal": 224}, "kultur": {"mapped": 80, "normal": 81}, "other": {"mapped": 7540, "normal": 7631}, "topic": {"mapped": 41, "normal": 42}, "sport": {"mapped": 2804, "normal": 2832}, "download": {"mapped": 0, "normal": 16}, "indland": {"mapped": 270, "normal": 271}, "aarhus": {"mapped": 79, "normal": 79}, "indblik": {"mapped": 162, "normal": 161}, "navne": {"mapped": 8, "normal": 8}, "article": {"mapped": 5, "normal": 5}, "international": {"mapped": 231, "normal": 230}, "timeout": {"mapped": 75, "normal": 75}, "viden": {"mapped": 1124, "normal": 1130}, "erhverv": {"mapped": 21, "normal": 22}}}, "Platform": {"total": {"Computer": {"mapped": 291821, "normal": 291821}, "Big screen": {"mapped": 29, "normal": 29}, "Mobile": {"mapped": 390598, "normal": 390598}, "Unspecified": {"mapped": 1899, "normal": 1899}, "Tablet": {"mapped": 222191, "normal": 222191}, "Unknown": {"mapped": 5, "normal": 5}}, "sso": {"Unspecified": {"mapped": 272, "normal": 228}, "Tablet": {"mapped": 61981, "normal": 39986}, "Mobile": {"mapped": 74023, "normal": 46439}, "Computer": {"mapped": 62596, "normal": 54292}, "Unknown": {"mapped": 4, "normal": 0}}, "primary": {"Unspecified": {"mapped": 26, "normal": 26}, "Tablet": {"mapped": 4648, "normal": 4782}, "Mobile": {"mapped": 4133, "normal": 4171}, "Computer": {"mapped": 5071, "normal": 5099}}}, "Web browser": {"total": {"UC Browser": {"mapped": 9, "normal": 9}, "Opera Mini": {"mapped": 133, "normal": 133}, "NetFront": {"mapped": 9, "normal": 9}, "Iceweasel": {"mapped": 2, "normal": 2}, "IE": {"mapped": 64221, "normal": 64221}, "Facebook for iPhone": {"mapped": 31056, "normal": 31056}, "Thunderbird": {"mapped": 6, "normal": 6}, "Vienna": {"mapped": 1, "normal": 1}, "SeaMonkey": {"mapped": 30, "normal": 30}, "Firefox": {"mapped": 40869, "normal": 40869}, "Opera": {"mapped": 65, "normal": 65}, "Chromium": {"mapped": 277, "normal": 277}, "Firefox for iOS": {"mapped": 791, "normal": 791}, "Chrome for iOS": {"mapped": 16323, "normal": 16323}, "Edge": {"mapped": 42148, "normal": 42148}, "Chrome": {"mapped": 289442, "normal": 289442}, "BlackBerry Browser": {"mapped": 7, "normal": 7}, "Maxthon": {"mapped": 76, "normal": 76}, "Android Browser": {"mapped": 897, "normal": 897}, "Amazon Silk": {"mapped": 60, "normal": 60}, "Facebook for Android": {"mapped": 6090, "normal": 6090}, "Safari": {"mapped": 413874, "normal": 413874}, "Vivaldi": {"mapped": 118, "normal": 118}, "Unknown": {"mapped": 19, "normal": 19}, "PaleMoon": {"mapped": 20, "normal": 20}}, "sso": {"Chromium": {"mapped": 30, "normal": 27}, "Firefox for iOS": {"mapped": 184, "normal": 111}, "Chrome for iOS": {"mapped": 4073, "normal": 3182}, "Edge": {"mapped": 9079, "normal": 7995}, "Chrome": {"mapped": 57731, "normal": 51231}, "Opera Mini": {"mapped": 113, "normal": 1}, "Facebook for Android": {"mapped": 174, "normal": 151}, "Maxthon": {"mapped": 6, "normal": 6}, "Android Browser": {"mapped": 317, "normal": 235}, "Amazon Silk": {"mapped": 21, "normal": 20}, "IE": {"mapped": 12019, "normal": 10190}, "Facebook for iPhone": {"mapped": 527, "normal": 372}, "Safari": {"mapped": 107514, "normal": 61282}, "Vivaldi": {"mapped": 4, "normal": 3}, "SeaMonkey": {"mapped": 2, "normal": 2}, "Unknown": {"mapped": 4, "normal": 0}, "Firefox": {"mapped": 7078, "normal": 6137}}, "primary": {"Chromium": {"mapped": 5, "normal": 5}, "Firefox for iOS": {"mapped": 13, "normal": 14}, "Chrome for iOS": {"mapped": 294, "normal": 301}, "Edge": {"mapped": 835, "normal": 839}, "Chrome": {"mapped": 3813, "normal": 3813}, "Opera Mini": {"mapped": 0, "normal": 1}, "Facebook for Android": {"mapped": 53, "normal": 54}, "Maxthon": {"mapped": 1, "normal": 1}, "Android Browser": {"mapped": 17, "normal": 18}, "Amazon Silk": {"mapped": 1, "normal": 1}, "IE": {"mapped": 1155, "normal": 1169}, "Facebook for iPhone": {"mapped": 121, "normal": 121}, "Safari": {"mapped": 6893, "normal": 7061}, "Vivaldi": {"mapped": 2, "normal": 2}, "SeaMonkey": {"mapped": 1, "normal": 1}, "Firefox": {"mapped": 674, "normal": 677}}}, "Operating system": {"total": {"FreeBSD": {"mapped": 3, "normal": 3}, "Xbox OS": {"mapped": 2, "normal": 2}, "Chromium OS": {"mapped": 1131, "normal": 1131}, "PlayStation": {"mapped": 9, "normal": 9}, "Unspecified": {"mapped": 1, "normal": 1}, "iOS": {"mapped": 423594, "normal": 423594}, "Tizen": {"mapped": 9, "normal": 9}, "Unknown": {"mapped": 12, "normal": 12}, "Mac OS": {"mapped": 55366, "normal": 55366}, "Windows": {"mapped": 232764, "normal": 232764}, "BlackBerry OS": {"mapped": 7, "normal": 7}, "Java": {"mapped": 66, "normal": 66}, "Symbian": {"mapped": 138, "normal": 138}, "Windows Phone": {"mapped": 12159, "normal": 12159}, "Android": {"mapped": 176823, "normal": 176823}, "Windows RT": {"mapped": 184, "normal": 184}, "Linux": {"mapped": 4275, "normal": 4275}}, "sso": {"iOS": {"mapped": 104373, "normal": 58434}, "Chromium OS": {"mapped": 302, "normal": 279}, "Windows RT": {"mapped": 19, "normal": 17}, "Unspecified": {"mapped": 1, "normal": 0}, "Linux": {"mapped": 615, "normal": 551}, "Mac OS": {"mapped": 12197, "normal": 10250}, "Windows": {"mapped": 49876, "normal": 43554}, "Java": {"mapped": 54, "normal": 0}, "Symbian": {"mapped": 14, "normal": 12}, "Windows Phone": {"mapped": 1996, "normal": 1778}, "Android": {"mapped": 29425, "normal": 26070}, "Unknown": {"mapped": 4, "normal": 0}}, "primary": {"Mac OS": {"mapped": 1076, "normal": 1081}, "Windows": {"mapped": 4538, "normal": 4566}, "Chromium OS": {"mapped": 25, "normal": 25}, "Windows RT": {"mapped": 1, "normal": 1}, "Symbian": {"mapped": 2, "normal": 2}, "Android": {"mapped": 1437, "normal": 1444}, "Windows Phone": {"mapped": 184, "normal": 182}, "iOS": {"mapped": 6580, "normal": 6741}, "Linux": {"mapped": 35, "normal": 36}}}};
    
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
    
    // sort items by value
    sorted.sort(function(a, b) {
	return b[1] - a[1]; // compare numbers
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
        .style("fill", function(d, i) { return (i < keys.length) ? 'rgba(204, 34, 20, 0.5)' : 'rgba(0, 0, 0, 0.5)'; });

    g.append("text")
        .attr("x", (width / 2))             
        .attr("y", 0 - (margin.top / 2))
        .attr("text-anchor", "middle")  
        .style("font-size", "16px") 
        .text(chartname);
};
