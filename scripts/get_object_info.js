// Max JavaScript to get object information
// This script can be run inside Max to extract object port information

autowatch = 1;
inlets = 1;
outlets = 1;

// List of objects to test
var objects_to_test = [
    "cycle~", "dac~", "adc~", "+", "-", "*", "/",
    "metro", "button", "toggle", "flonum", "number",
    "message", "pack", "unpack", "route", "select",
    "notein", "noteout", "ctlin", "ctlout"
];

function bang() {
    var results = {};
    
    for (var i = 0; i < objects_to_test.length; i++) {
        var obj_name = objects_to_test[i];
        try {
            // Create object in patcher
            var obj = this.patcher.newdefault(100, 100 + i * 30, obj_name);
            
            if (obj) {
                var info = {
                    maxclass: obj.maxclass,
                    numinlets: obj.getattr("numinlets") || 0,
                    numoutlets: obj.getattr("numoutlets") || 0
                };
                
                // Try to get more detailed information
                try {
                    // Get inlet assists (descriptions)
                    var inlet_assists = [];
                    for (var j = 0; j < info.numinlets; j++) {
                        var assist = obj.assist("inlet", j);
                        if (assist) inlet_assists.push(assist);
                    }
                    info.inlet_assists = inlet_assists;
                    
                    // Get outlet assists
                    var outlet_assists = [];
                    for (var j = 0; j < info.numoutlets; j++) {
                        var assist = obj.assist("outlet", j);
                        if (assist) outlet_assists.push(assist);
                    }
                    info.outlet_assists = outlet_assists;
                } catch(e) {
                    // Some objects might not support assist
                }
                
                results[obj_name] = info;
                
                // Remove the object
                this.patcher.remove(obj);
            }
        } catch(e) {
            post("Error creating " + obj_name + ": " + e + "\n");
        }
    }
    
    // Output results
    outlet(0, "dictionary", JSON.stringify(results, null, 2));
}

// Test individual object
function test(obj_name) {
    try {
        var obj = this.patcher.newdefault(100, 100, obj_name);
        
        if (obj) {
            post(obj_name + " info:\n");
            post("  maxclass: " + obj.maxclass + "\n");
            post("  numinlets: " + (obj.getattr("numinlets") || 0) + "\n");
            post("  numoutlets: " + (obj.getattr("numoutlets") || 0) + "\n");
            
            // List all available attributes
            var attrs = obj.getattrnames();
            if (attrs) {
                post("  attributes: " + attrs.join(", ") + "\n");
            }
            
            this.patcher.remove(obj);
        }
    } catch(e) {
        post("Error: " + e + "\n");
    }
}