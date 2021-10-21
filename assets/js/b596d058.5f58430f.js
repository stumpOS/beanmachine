"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[9775],{3905:function(e,n,r){r.r(n),r.d(n,{MDXContext:function(){return l},MDXProvider:function(){return m},mdx:function(){return f},useMDXComponents:function(){return d},withMDXComponents:function(){return p}});var t=r(67294);function a(e,n,r){return n in e?Object.defineProperty(e,n,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[n]=r,e}function i(){return i=Object.assign||function(e){for(var n=1;n<arguments.length;n++){var r=arguments[n];for(var t in r)Object.prototype.hasOwnProperty.call(r,t)&&(e[t]=r[t])}return e},i.apply(this,arguments)}function o(e,n){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);n&&(t=t.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),r.push.apply(r,t)}return r}function s(e){for(var n=1;n<arguments.length;n++){var r=null!=arguments[n]?arguments[n]:{};n%2?o(Object(r),!0).forEach((function(n){a(e,n,r[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(r,n))}))}return e}function c(e,n){if(null==e)return{};var r,t,a=function(e,n){if(null==e)return{};var r,t,a={},i=Object.keys(e);for(t=0;t<i.length;t++)r=i[t],n.indexOf(r)>=0||(a[r]=e[r]);return a}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(t=0;t<i.length;t++)r=i[t],n.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var l=t.createContext({}),p=function(e){return function(n){var r=d(n.components);return t.createElement(e,i({},n,{components:r}))}},d=function(e){var n=t.useContext(l),r=n;return e&&(r="function"==typeof e?e(n):s(s({},n),e)),r},m=function(e){var n=d(e.components);return t.createElement(l.Provider,{value:n},e.children)},u={inlineCode:"code",wrapper:function(e){var n=e.children;return t.createElement(t.Fragment,{},n)}},h=t.forwardRef((function(e,n){var r=e.components,a=e.mdxType,i=e.originalType,o=e.parentName,l=c(e,["components","mdxType","originalType","parentName"]),p=d(r),m=a,h=p["".concat(o,".").concat(m)]||p[m]||u[m]||i;return r?t.createElement(h,s(s({ref:n},l),{},{components:r})):t.createElement(h,s({ref:n},l))}));function f(e,n){var r=arguments,a=n&&n.mdxType;if("string"==typeof e||a){var i=r.length,o=new Array(i);o[0]=h;var s={};for(var c in n)hasOwnProperty.call(n,c)&&(s[c]=n[c]);s.originalType=e,s.mdxType="string"==typeof e?e:a,o[1]=s;for(var l=2;l<i;l++)o[l]=r[l];return t.createElement.apply(null,o)}return t.createElement.apply(null,r)}h.displayName="MDXCreateElement"},38060:function(e,n,r){r.r(n),r.d(n,{frontMatter:function(){return s},contentTitle:function(){return c},metadata:function(){return l},toc:function(){return p},default:function(){return m}});var t=r(87462),a=r(63366),i=(r(67294),r(3905)),o=["components"],s={id:"bmg",title:"Bean Machine Graph (BMG)",sidebar_label:"BMG"},c=void 0,l={unversionedId:"overview/bmg/bmg",id:"overview/bmg/bmg",isDocsHomePage:!1,title:"Bean Machine Graph (BMG)",description:"Bean Machine Graph (BMG) is an experimental C++ inference engine optimized for MCMC inference on static graphical models (i.e., models whose dependency structure is independent of all random variables\u2019 values). Whereas Bean Machine models are defined through a series of random variables and functionals, Bean Machine Graph models are defined through the explicit construction of their computation graphs. Bean Machine Graph then performs inference directly on this graphical representation of the model.",source:"@site/../docs/overview/bmg/bmg.md",sourceDirName:"overview/bmg",slug:"/overview/bmg/bmg",permalink:"/docs/overview/bmg/bmg",editUrl:"https://github.com/facebook/docusaurus/edit/master/website/../docs/overview/bmg/bmg.md",tags:[],version:"current",frontMatter:{id:"bmg",title:"Bean Machine Graph (BMG)",sidebar_label:"BMG"},sidebar:"someSidebar",previous:{title:"transforms",permalink:"/docs/framework_topics/programmable_inference/transforms"},next:{title:"Beanstalk",permalink:"/docs/overview/beanstalk/beanstalk"}},p=[{value:"Modeling",id:"modeling",children:[],level:2},{value:"Inference",id:"inference",children:[],level:2},{value:"Using Bean Machine Graph",id:"using-bean-machine-graph",children:[],level:2}],d={toc:p};function m(e){var n=e.components,s=(0,a.Z)(e,o);return(0,i.mdx)("wrapper",(0,t.Z)({},d,s,{components:n,mdxType:"MDXLayout"}),(0,i.mdx)("p",null,"Bean Machine Graph (BMG) is an experimental C++ inference engine optimized for MCMC inference on static graphical models (i.e., models whose dependency structure is independent of all random variables\u2019 values). Whereas Bean Machine models are defined through a series of random variables and functionals, Bean Machine Graph models are defined through the explicit construction of their computation graphs. Bean Machine Graph then performs inference directly on this graphical representation of the model."),(0,i.mdx)("h2",{id:"modeling"},"Modeling"),(0,i.mdx)("p",null,"Bean Machine Graph currently supports models with static graphs, as the full computation graph for the model must be explicitly defined at compile-time. The graphical representation of a model consists of a collection of nodes representing distributions, samples, constants, operators, and so on. For example, consider the following model:"),(0,i.mdx)("pre",null,(0,i.mdx)("code",{parentName:"pre"},"@random_variable\ndef a():\n    return Normal(0, 1.)\n\n@random_variable\ndef b():\n    return Normal(a() + 5, 1.)\n")),(0,i.mdx)("p",null,"In Bean Machine Graph, we represent it with the following graph:\n",(0,i.mdx)("img",{alt:"Typical DOT rendering of graph for model above",src:r(22933).Z})),(0,i.mdx)("p",null,"Though Bean Machine Graph supports many commonly-used distributions and operators, it is less expressive than Bean Machine Python, which supports dynamic models, rich programmable inference, and so on."),(0,i.mdx)("h2",{id:"inference"},"Inference"),(0,i.mdx)("p",null,"Bean Machine Graph currently supports Newtonian Monte Carlo, Gibbs sampling for Boolean variables, as well as rejection sampling. It performs inference efficiently by taking advantage of the static nature of its models. It also uses its own MCMC-focused automatic differentiation (AD) engine and performs the necessary gradient computations faster than other general-purpose AD engines which often have additional overhead."),(0,i.mdx)("h2",{id:"using-bean-machine-graph"},"Using Bean Machine Graph"),(0,i.mdx)("p",null,"Due to the complexity of constructing the computation graph directly, users interested in Bean Machine Graph should use the ",(0,i.mdx)("a",{parentName:"p",href:"../beanstalk/beanstalk"},"Beanstalk")," compiler, which enables users to write models using Bean Machine Python syntax and call the Bean Machine Graph inference engine as if it were another Bean Machine inference method.\nTo learn more about the current state of inference in Bean Machine Graph and experiment with different models, see the ",(0,i.mdx)("a",{parentName:"p",href:"../beanstalk/beanstalk"},"Beanstalk documentation"),"."))}m.isMDXComponent=!0},22933:function(e,n,r){n.Z=r.p+"assets/images/image-dbb3a3a615eddfd4d377ee696ba54155.png"}}]);