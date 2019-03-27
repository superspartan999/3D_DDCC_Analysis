 


///////
//// GenerateByMathematicaCode /////
//// Lheureux2018 /////
///////


la=2*10^-6;//10nm
u=10^-4;//um
n=10^-7;//nm
a=10^-8;//a
length = 30 *n; //nm
mesh_x = 0.6 * n;
tf = length/mesh_x;

tf_qw = 15;
tf_cap = 10;
////////////////////////////////////////////
//// SUBSTRATE LAYER ///////
////////////////////////////////////////////
 

//// Points Definition ///////

p1=newp;Point(p1)={0,0,0,la};
p2=newp;Point(p2)={length,0,0,la};
p3=newp;Point(p3)={length,length,0,la};
p4=newp;Point(p4)={0,length,0,la};

//// Line Definition ///////

l1=newl;Line(l1)={p1,p2};
l2=newl;Line(l2)={p2,p3};
l3=newl;Line(l3)={p3,p4};
l4=newl;Line(l4)={p4,p1};

ll1=newll;Line Loop(ll1)={l1,l2,l3,l4};
ps1=news;Plane Surface(ps1)={ll1};

//// Mesh Definition ///////

Transfinite Line{l1}=tf+1;
Transfinite Line{l2}=tf+1;
Transfinite Line{l3}=tf+1;
Transfinite Line{l4}=tf+1;

Transfinite Surface{ps1}={1,2,3,4};



///////////////////////////////////////////
//*****NEW VOL 2***  ///////
///////////////////////////////////////////

//// Points Definition ///////

p11=newp;Point(p11)={0,0,20.*n,la};
p12=newp;Point(p12)={length,0,20.*n,la};
p13=newp;Point(p13)={length,length,20.*n,la};
p14=newp;Point(p14)={0,length,20.*n,la};

//// Definition des lignes du Volume

l11=newl;Line(l11)={p11,p12};
l12=newl;Line(l12)={p12,p13};
l13=newl;Line(l13)={p13,p14};
l14=newl;Line(l14)={p14,p11};
l15=newl;Line(l15)={p1,p11};
l16=newl;Line(l16)={p2,p12};
l17=newl;Line(l17)={p3,p13};
l18=newl;Line(l18)={p4,p14};

//////////////////////////
//// Mesh Definition of Volume 2 ///////
//////////////////////////

Transfinite Line{l11}= tf+1;
Transfinite Line{l12}=tf+1;
Transfinite Line{l13}=tf+1;
Transfinite Line{l14}= tf+1;
Transfinite Line{l15}=37 Using Progression 1.08;
Transfinite Line{l16}=37 Using Progression 1.08;
Transfinite Line{l17}=37 Using Progression 1.08;
Transfinite Line{l18}=37 Using Progression 1.08;

ll11=newll;Line Loop(ll11)={l11,l12,l13,l14};
ll12=newll;Line Loop(ll12)={l1,l16,-l11,-l15};
ll13=newll;Line Loop(ll13)={l2,l17,-l12,-l16};
ll14=newll;Line Loop(ll14)={l3,l18,-l13,-l17};
ll15=newll;Line Loop(ll15)={l4,l15,-l14,-l18};

ps11=news;Plane Surface(ps11)={ll11};
ps12=news;Plane Surface(ps12)={ll12};
ps13=news;Plane Surface(ps13)={ll13};
ps14=news;Plane Surface(ps14)={ll14};
ps15=news;Plane Surface(ps15)={ll15};

Transfinite Surface{ps11}={p11,p12,p13,p14};
Transfinite Surface{ps12}={p1,p2,p12,p11};
Transfinite Surface{ps13}={p2,p3,p13,p12};
Transfinite Surface{ps14}={p4,p3,p13,p14};
Transfinite Surface{ps15}={p1,p4,p14,p11};

Surface Loop(1)={ps1,ps11,ps12,ps13,ps14,ps15};
Volume(1)={1};
Transfinite Volume(1);

