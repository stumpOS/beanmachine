"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[4441],{3905:function(e,t,n){n.r(t),n.d(t,{MDXContext:function(){return u},MDXProvider:function(){return d},mdx:function(){return h},useMDXComponents:function(){return m},withMDXComponents:function(){return c}});var a=n(67294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function s(){return s=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var a in n)Object.prototype.hasOwnProperty.call(n,a)&&(e[a]=n[a])}return e},s.apply(this,arguments)}function r(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?r(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):r(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,a,i=function(e,t){if(null==e)return{};var n,a,i={},s=Object.keys(e);for(a=0;a<s.length;a++)n=s[a],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var s=Object.getOwnPropertySymbols(e);for(a=0;a<s.length;a++)n=s[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var u=a.createContext({}),c=function(e){return function(t){var n=m(t.components);return a.createElement(e,s({},t,{components:n}))}},m=function(e){var t=a.useContext(u),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},d=function(e){var t=m(e.components);return a.createElement(u.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},f=a.forwardRef((function(e,t){var n=e.components,i=e.mdxType,s=e.originalType,r=e.parentName,u=l(e,["components","mdxType","originalType","parentName"]),c=m(n),d=i,f=c["".concat(r,".").concat(d)]||c[d]||p[d]||s;return n?a.createElement(f,o(o({ref:t},u),{},{components:n})):a.createElement(f,o({ref:t},u))}));function h(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var s=n.length,r=new Array(s);r[0]=f;var o={};for(var l in t)hasOwnProperty.call(t,l)&&(o[l]=t[l]);o.originalType=e,o.mdxType="string"==typeof e?e:i,r[1]=o;for(var u=2;u<s;u++)r[u]=n[u];return a.createElement.apply(null,r)}return a.createElement.apply(null,n)}f.displayName="MDXCreateElement"},21847:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return o},contentTitle:function(){return l},metadata:function(){return u},toc:function(){return c},default:function(){return d}});var a=n(87462),i=n(63366),s=(n(67294),n(3905)),r=["components"],o={},l="Model Evaluation in Bean Machine: Diagnostics Module",u={unversionedId:"framework_topics/model_evaluation/diagnostics",id:"framework_topics/model_evaluation/diagnostics",isDocsHomePage:!1,title:"Model Evaluation in Bean Machine: Diagnostics Module",description:"This notebook introduces the Diagnostics class in Bean Machine (BM) which aims to assist the modeler to get insights about the model performance. Diagnostics currently supports two main components: 1) General Summary Statistics module and 2) Visualizer module.",source:"@site/../docs/framework_topics/model_evaluation/diagnostics.md",sourceDirName:"framework_topics/model_evaluation",slug:"/framework_topics/model_evaluation/diagnostics",permalink:"/docs/framework_topics/model_evaluation/diagnostics",editUrl:"https://github.com/facebook/docusaurus/edit/master/website/../docs/framework_topics/model_evaluation/diagnostics.md",tags:[],version:"current",frontMatter:{},sidebar:"someSidebar",previous:{title:"Logging",permalink:"/docs/framework_topics/development/logging"},next:{title:"Model comparison",permalink:"/docs/framework_topics/model_evaluation/model_comparison"}},c=[{value:"1. Getting the full statistics summary of all queries over all chains",id:"1-getting-the-full-statistics-summary-of-all-queries-over-all-chains",children:[],level:2},{value:"2. Getting summary statistics for a subset of queries",id:"2-getting-summary-statistics-for-a-subset-of-queries",children:[],level:2},{value:"3. Getting summary for a specific chain",id:"3-getting-summary-for-a-specific-chain",children:[],level:2},{value:"4. Extend summary module by new functions:",id:"4-extend-summary-module-by-new-functions",children:[],level:2},{value:"5. Individual calling of summary statistics functions:",id:"5-individual-calling-of-summary-statistics-functions",children:[],level:2},{value:"6. Override an already registered function",id:"6-override-an-already-registered-function",children:[],level:2},{value:"1. Execute all plot-related functions for all queries",id:"1-execute-all-plot-related-functions-for-all-queries",children:[],level:2},{value:"2. Execute all plot-related functions for a subset of queries",id:"2-execute-all-plot-related-functions-for-a-subset-of-queries",children:[],level:2},{value:"3. Update and display the plotly object",id:"3-update-and-display-the-plotly-object",children:[],level:2},{value:"4. Execute all plot-related functions for a specific chain",id:"4-execute-all-plot-related-functions-for-a-specific-chain",children:[],level:2},{value:"5. Individual calling of a plot-related function:",id:"5-individual-calling-of-a-plot-related-function",children:[],level:2}],m={toc:c};function d(e){var t=e.components,n=(0,i.Z)(e,r);return(0,s.mdx)("wrapper",(0,a.Z)({},m,n,{components:t,mdxType:"MDXLayout"}),(0,s.mdx)("h1",{id:"model-evaluation-in-bean-machine-diagnostics-module"},"Model Evaluation in Bean Machine: Diagnostics Module"),(0,s.mdx)("p",null,"This notebook introduces the Diagnostics class in Bean Machine (BM) which aims to assist the modeler to get insights about the model performance. Diagnostics currently supports two main components: 1) General Summary Statistics module and 2) Visualizer module."),(0,s.mdx)("p",null,(0,s.mdx)("strong",{parentName:"p"},"General Summary Statistics module")," aggregates the statistics of all or a subset of queries over a specific chain or all chains."),(0,s.mdx)("p",null,(0,s.mdx)("strong",{parentName:"p"},"Visualizer module")," processes samples and encapsulates the results in a plotly object which could be used for actual visualization."),(0,s.mdx)("p",null,"Both of the BM Diagnostics components support function registration which allows the user to extend each component with new functionalities that modeler might be interested to have."),(0,s.mdx)("p",null,"The rest of this document goes over examples of how each component can be called or extended."),(0,s.mdx)("h1",{id:"model-definition-and-running-inference"},"Model definition and running inference:"),(0,s.mdx)("p",null,"Suppose we want to use BM Diagnostics module over the inferred samples of the following model:"),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-python"},"Example:\n\n@sample\ndef dirichlet(i, j):\n    return dist.Dirichlet(\n    torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [2.0, 3.0, 1.0]])\n)\n\n\n@sample\ndef beta(i):\n    return dist.Beta(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([9.0, 8.0, 7.0]))\n\n\n@sample\ndef normal():\n    return dist.Normal(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([0.5, 1.0, 1.5]))\n\n\nmh = SingleSiteAncestralMetropolisHastings()\nchains = 2\nsamples = mh.infer([beta(0), dirichlet(1, 5), normal()], {}, 50, chains)\n")),(0,s.mdx)("h1",{id:"summary-stats"},"Summary Stats"),(0,s.mdx)("p",null,"Calling the summary() function on learned model parameters outputs a table including mean, standard deviation (std), confidence interval (CI), r-hat and effective sample size (n-eff)."),(0,s.mdx)("p",null,"User also has the flexibility to define new functions and register them as part of the available summary stat functions. So, when summary() is called, the results for the user defined function will be added to the output table."),(0,s.mdx)("p",null,"Here are different ways that modeler can call summary() function:"),(0,s.mdx)("h2",{id:"1-getting-the-full-statistics-summary-of-all-queries-over-all-chains"},"1. Getting the full statistics summary of all queries over all chains"),(0,s.mdx)("p",null,"Simply call summary() to get a comprehensive table of gathered statistics for all queries."),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-python"},"out_df= Diagnostics(samples).summary()\n")),(0,s.mdx)("h2",{id:"2-getting-summary-statistics-for-a-subset-of-queries"},"2. Getting summary statistics for a subset of queries"),(0,s.mdx)("p",null,"Considering a big list of queries that model may have, user may hand-pick a subset of queries to confine the output table."),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-python"},"Example:\n\nout_df = Diagnostics(samples).summary([dirichlet(1, 5), beta(0)])\n")),(0,s.mdx)("h2",{id:"3-getting-summary-for-a-specific-chain"},"3. Getting summary for a specific chain"),(0,s.mdx)("p",null,"To compare how results may change over the course of chains, user can pass the chain number to summary() function."),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-python"},"Example:\n\nout_df = Diagnostics(samples).summary(query_list=[dirichlet(1, 5)], chain=1)\n\n")),(0,s.mdx)("h2",{id:"4-extend-summary-module-by-new-functions"},"4. Extend summary module by new functions:"),(0,s.mdx)("p",null,"User has the option to extend the summary module by registering new functions or overwrite an already available functions. To add new functions, user should make a derived class of the Diagnostics class and register new functions in the class constructor as follow:"),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-python"},'\ndef newfunc(query_samples: Tensor) -> Tensor\n\nclass Mydiag(Diagnostics):\n    def __init__(self, samples: MonteCarloSamples):\n        super().__init__(samples)\n        self.newfunc = self.summaryfn(newfunc, display_names=["new_func"])\n')),(0,s.mdx)("p",null,(0,s.mdx)("strong",{parentName:"p"},"summaryfn")," wrapper treats the newfunc as part of the summary module and adds its results to the output table."),(0,s.mdx)("p",null,"The input to all default summary stat functions has the shape of ",(0,s.mdx)("strong",{parentName:"p"},"torch.Size(","[C, N]",")")," where ",(0,s.mdx)("strong",{parentName:"p"},"C")," and ",(0,s.mdx)("strong",{parentName:"p"},"N")," are the number of chains and samples-per-chain respectively considering that ",(0,s.mdx)("strong",{parentName:"p"},"query_samples")," is as follow:"),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-python"},"samples = mh.infer(...)\nquery_samples = samples[query] ## query_samples.shape = torch.Size([C, N])\n")),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-python"},'Example:\n\ndef mymean(query_samples: Tensor) -> Tensor:\n    return torch.mean(query_samples, dim=[0, 1])\n\n\nclass Mydiag(Diagnostics):\n    def __init__(self, samples: MonteCarloSamples):\n        super().__init__(samples)\n        self.mymean = self.summaryfn(mymean, display_names=["mymean"])\n\n\ncustomDiag = Mydiag(samples)\nout = customDiag.summary(query_list=[dirichlet(1, 5)], chain=0)\nout.head()\n')),(0,s.mdx)("h2",{id:"5-individual-calling-of-summary-statistics-functions"},"5. Individual calling of summary statistics functions:"),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-python"},"Example:\n\n# calling user-defined func over all chains\nout_df = customDiag.mymean([dirichlet(1, 5)])\n\n# calling user-defined func over a specific chain\nout_df = customDiag.mymean([dirichlet(1, 5)], chain = 1)\n\n# calling a default func over a specific chain\nout_df = customDiag.std([dirichlet(1, 5)], chain = 1)\n\n")),(0,s.mdx)("h2",{id:"6-override-an-already-registered-function"},"6. Override an already registered function"),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-python"},'Example:\n\ndef mean(query_samples: Tensor) -> Tensor:\n    y = query_samples + torch.ones(query_samples.size())\n    return torch.mean(y, dim=[0, 1])\n\nclass Mydiag(Diagnostics):\n    def __init__(self, samples: MonteCarloSamples):\n        super().__init__(samples)\n        self.mean = self.summaryfn(mean, display_names=["avg"])\n\ncustomDiag = Mydiag(samples)\nout = customDiag.summary(query_list=[dirichlet(1, 5)], chain=0)\n')),(0,s.mdx)("h1",{id:"visualization"},"Visualization:"),(0,s.mdx)("p",null,"Currently we support trace plots and auto-correlation plots for samples of a requested model parameter. User can define new visualization-related functions and register them via ",(0,s.mdx)("strong",{parentName:"p"},"plotfn")," wrapper. Each of these functions return a plotly object and so user can modify the object as he/she wishes. Here are different ways to call plot over whole or a subset of queries."),(0,s.mdx)("h2",{id:"1-execute-all-plot-related-functions-for-all-queries"},"1. Execute all plot-related functions for all queries"),(0,s.mdx)("p",null,"User can enable plotting the returned plotly object by passing display = True to the plot() function. The default is false which means that only the plotly object is returned. So, user has the flexibility to update the layout for the outputted object and frame the outputted plot they way he/she wishes."),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-python"},"Example:\nfig = Diagnostics(samples).plot()\n")),(0,s.mdx)("h2",{id:"2-execute-all-plot-related-functions-for-a-subset-of-queries"},"2. Execute all plot-related functions for a subset of queries"),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-python"},"Example:\nfigs = Diagnostics(samples).plot(query_list=[dirichlet(1, 5)])\n")),(0,s.mdx)("h2",{id:"3-update-and-display-the-plotly-object"},"3. Update and display the plotly object"),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-python"},"Example:\nfor _i,fig in enumerate(figs):\n    fig.update_layout(paper_bgcolor=\"LightBlue\",height=1500, width=700,)\n    fig.update_layout(legend_orientation=\"h\")\n    fig.update_xaxes(title_font=dict(size=14, family='Courier', color='crimson'))\n    fig.update_yaxes(title_font=dict(size=14, family='Courier', color='crimson'))\n    plotly.offline.iplot(fig)\n")),(0,s.mdx)("h2",{id:"4-execute-all-plot-related-functions-for-a-specific-chain"},"4. Execute all plot-related functions for a specific chain"),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-python"},"Example:\nd = Diagnostics(samples)\nfigs = d.plot(query_list=[dirichlet(1, 5)], chain = 0)\n")),(0,s.mdx)("h2",{id:"5-individual-calling-of-a-plot-related-function"},"5. Individual calling of a plot-related function:"),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-python"},'Example:\nd = Diagnostics(samples)\n\nautocorr_object = d.autocorr([dirichlet(1, 5)]) # pass "display = True" to output the plot\nautocorr_object = d.autocorr([dirichlet(1, 5)], chain = 0)\n')))}d.isMDXComponent=!0}}]);