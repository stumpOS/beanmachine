"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[8858],{3905:function(e,t,r){r.r(t),r.d(t,{MDXContext:function(){return p},MDXProvider:function(){return m},mdx:function(){return y},useMDXComponents:function(){return s},withMDXComponents:function(){return l}});var n=r(67294);function o(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function c(){return c=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var r=arguments[t];for(var n in r)Object.prototype.hasOwnProperty.call(r,n)&&(e[n]=r[n])}return e},c.apply(this,arguments)}function i(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function a(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?i(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):i(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function u(e,t){if(null==e)return{};var r,n,o=function(e,t){if(null==e)return{};var r,n,o={},c=Object.keys(e);for(n=0;n<c.length;n++)r=c[n],t.indexOf(r)>=0||(o[r]=e[r]);return o}(e,t);if(Object.getOwnPropertySymbols){var c=Object.getOwnPropertySymbols(e);for(n=0;n<c.length;n++)r=c[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var p=n.createContext({}),l=function(e){return function(t){var r=s(t.components);return n.createElement(e,c({},t,{components:r}))}},s=function(e){var t=n.useContext(p),r=t;return e&&(r="function"==typeof e?e(t):a(a({},t),e)),r},m=function(e){var t=s(e.components);return n.createElement(p.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},f=n.forwardRef((function(e,t){var r=e.components,o=e.mdxType,c=e.originalType,i=e.parentName,p=u(e,["components","mdxType","originalType","parentName"]),l=s(r),m=o,f=l["".concat(i,".").concat(m)]||l[m]||d[m]||c;return r?n.createElement(f,a(a({ref:t},p),{},{components:r})):n.createElement(f,a({ref:t},p))}));function y(e,t){var r=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var c=r.length,i=new Array(c);i[0]=f;var a={};for(var u in t)hasOwnProperty.call(t,u)&&(a[u]=t[u]);a.originalType=e,a.mdxType="string"==typeof e?e:o,i[1]=a;for(var p=2;p<c;p++)i[p]=r[p];return n.createElement.apply(null,i)}return n.createElement.apply(null,r)}f.displayName="MDXCreateElement"},79613:function(e,t,r){r.r(t),r.d(t,{frontMatter:function(){return a},contentTitle:function(){return u},metadata:function(){return p},toc:function(){return l},Highlight:function(){return s},default:function(){return d}});var n=r(87462),o=r(63366),c=(r(67294),r(3905)),i=["components"],a={id:"mdx",title:"Powered by MDX"},u=void 0,p={unversionedId:"mdx",id:"mdx",isDocsHomePage:!1,title:"Powered by MDX",description:"You can write JSX and use React components within your Markdown thanks to MDX.",source:"@site/../docs/mdx.md",sourceDirName:".",slug:"/mdx",permalink:"/docs/mdx",editUrl:"https://github.com/facebook/docusaurus/edit/master/website/../docs/mdx.md",tags:[],version:"current",frontMatter:{id:"mdx",title:"Powered by MDX"}},l=[],s=function(e){var t=e.children,r=e.color;return(0,c.mdx)("span",{style:{backgroundColor:r,borderRadius:"2px",color:"#fff",padding:"0.2rem"}},t)},m={toc:l,Highlight:s};function d(e){var t=e.components,r=(0,o.Z)(e,i);return(0,c.mdx)("wrapper",(0,n.Z)({},m,r,{components:t,mdxType:"MDXLayout"}),(0,c.mdx)("p",null,"You can write JSX and use React components within your Markdown thanks to ",(0,c.mdx)("a",{parentName:"p",href:"https://mdxjs.com/"},"MDX"),"."),(0,c.mdx)(s,{color:"#25c2a0",mdxType:"Highlight"},"Docusaurus green")," and ",(0,c.mdx)(s,{color:"#1877F2",mdxType:"Highlight"},"Facebook blue")," are my favorite colors.",(0,c.mdx)("p",null,"I can write ",(0,c.mdx)("strong",{parentName:"p"},"Markdown")," alongside my ",(0,c.mdx)("em",{parentName:"p"},"JSX"),"!"))}d.isMDXComponent=!0}}]);