/////GeneratedbyPython
/////ClaytonQwah2018

la = 2*10^-6; // 10 nm
u = 10^-4; // um 
n = 10^-7; // nm 
a = 10^-8; // a 
length = 30 *n; // nm 
mesh_x = 0.6 * n; 
tf = length/mesh_x; 

tf_qw = 15; 
tf_cap = 10; 
// // // // // // // // // // // // // // // // // // // // // // 
                    // // SUBSTRATE LAYER // // // / 
// // // // //// // // // // // // // // // // // // // // // // 
/////GeneratedbyPython
/////ClaytonQwah2018

la = 2*10^-6; // 10 nm
u = 10^-4; // um 
n = 10^-7; // nm 
a = 10^-8; // a 
length = 30 *n; // nm 
mesh_x = 0.6 * n; 
tf = length/mesh_x; 

tf_qw = 15; 
tf_cap = 10; 
// // // // // // // // // // // // // // // // // // // // // // 
                    // // SUBSTRATE LAYER // // // / 
// // // // //// // // // // // // // // // // // // // // // // 

// // Points Definition // // // / 

p1 = newp; Point (p1) = {0, 0, 0, la};
p2 = newp; Point (p2) = {length, 0, 0, la};
p3 = newp; Point (p3) = {length, length, 0, la};
p4 = newp; Point (p4) = {0, length, 0, la};

// // Line Definition // // // /

l1 = newl; Line (l1) = {p1, p2};
l2 = newl; Line (l2) = {p2, p3};
l3 = newl; Line (l3) = {p3, p4};
l4 = newl; Line (l4) = {p4, p1};

ll1 = newll; Line Loop (ll1) = {l1, l2, l3, l4};
ps1 = news; Plane Surface (ps1) = {ll1};

// // Mesh Definition // // // /

Transfinite Line {l1} = tf + 1;
Transfinite Line {l2} = tf + 1;
Transfinite Line {l3} = tf + 1;
Transfinite Line {l4} = tf + 1;

Transfinite Surface {ps1} = {1, 2, 3, 4};/////GeneratedbyPython
/////ClaytonQwah2018

la = 2*10^-6; // 10 nm
u = 10^-4; // um 
n = 10^-7; // nm 
a = 10^-8; // a 
length = 30 *n; // nm 
mesh_x = 0.6 * n; 
tf = length/mesh_x; 

tf_qw = 15; 
tf_cap = 10; 
// // // // // // // // // // // // // // // // // // // // // // 
                    // // SUBSTRATE LAYER // // // / 
// // // // //// // // // // // // // // // // // // // // // // 

// // Points Definition // // // / 

p1 = newp; Point (p1) = {0, 0, 0, la};
p2 = newp; Point (p2) = {length, 0, 0, la};
p3 = newp; Point (p3) = {length, length, 0, la};
p4 = newp; Point (p4) = {0, length, 0, la};

// // Line Definition // // // /

l1 = newl; Line (l1) = {p1, p2};
l2 = newl; Line (l2) = {p2, p3};
l3 = newl; Line (l3) = {p3, p4};
l4 = newl; Line (l4) = {p4, p1};

ll1 = newll; Line Loop (ll1) = {l1, l2, l3, l4};
ps1 = news; Plane Surface (ps1) = {ll1};

// // Mesh Definition // // // /

Transfinite Line {l1} = tf + 1;
Transfinite Line {l2} = tf + 1;
Transfinite Line {l3} = tf + 1;
Transfinite Line {l4} = tf + 1;

Transfinite Surface {ps1} = {1, 2, 3, 4};