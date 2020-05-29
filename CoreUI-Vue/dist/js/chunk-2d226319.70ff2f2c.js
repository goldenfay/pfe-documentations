(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-2d226319"],{e82b:function(t,s,r){"use strict";r.r(s);var e=function(){var t=this,s=t.$createElement,r=t._self._c||s;return r("CRow",[r("CCol",{attrs:{col:"12",md:"6"}},[r("CCard",[r("CCardHeader",[r("CIcon",{attrs:{name:"cil-justify-center"}}),r("strong",[t._v(" Bootstrap Alert")]),r("div",{staticClass:"card-header-actions"},[r("a",{staticClass:"card-header-action",attrs:{href:"https://coreui.io/vue/docs/components/alert",rel:"noreferrer noopener",target:"_blank"}},[r("small",{staticClass:"text-muted"},[t._v("docs")])])])],1),r("CCardBody",[r("div",[r("p"),r("CAlert",{attrs:{show:"",color:"primary"}},[t._v("Primary Alert")]),r("CAlert",{attrs:{show:"",color:"secondary"}},[t._v("Secondary Alert")]),r("CAlert",{attrs:{show:"",color:"success"}},[t._v("Success Alert")]),r("CAlert",{attrs:{show:"",color:"danger"}},[t._v("Danger Alert")]),r("CAlert",{attrs:{show:"",color:"warning"}},[t._v("Warning Alert")]),r("CAlert",{attrs:{show:"",color:"info"}},[t._v("Info Alert")]),r("CAlert",{attrs:{show:"",color:"light"}},[t._v("Light Alert")]),r("CAlert",{attrs:{show:"",color:"dark"}},[t._v("Dark Alert")])],1)])],1)],1),r("CCol",{attrs:{col:"12",md:"6"}},[r("CCard",[r("CCardHeader",[r("CIcon",{attrs:{name:"cil-justify-center"}}),t._v(" Alert "),r("small",[t._v(" use "),r("code",[t._v(".alert-link")]),t._v(" to provide links")])],1),r("CCardBody",[r("div",[r("CAlert",{attrs:{show:"",color:"primary"}},[t._v(" Primary Alert with "),r("a",{staticClass:"alert-link",attrs:{href:"#"}},[t._v("an example link")]),t._v(". ")]),r("CAlert",{attrs:{show:"",color:"secondary"}},[t._v(" Secondary Alert with "),r("a",{staticClass:"alert-link",attrs:{href:"#"}},[t._v("an example link")]),t._v(". ")]),r("CAlert",{attrs:{show:"",color:"success"}},[t._v(" Success Alert with "),r("a",{staticClass:"alert-link",attrs:{href:"#"}},[t._v("an example link")]),t._v(". ")]),r("CAlert",{attrs:{show:"",color:"danger"}},[t._v(" Danger Alert with "),r("a",{staticClass:"alert-link",attrs:{href:"#"}},[t._v("an example link")]),t._v(". ")]),r("CAlert",{attrs:{show:"",color:"warning"}},[t._v(" Warning Alert with "),r("a",{staticClass:"alert-link",attrs:{href:"#"}},[t._v("an example link")]),t._v(". ")]),r("CAlert",{attrs:{show:"",color:"info"}},[t._v(" Info Alert with "),r("a",{staticClass:"alert-link",attrs:{href:"#"}},[t._v("an example link")]),t._v(". ")]),r("CAlert",{attrs:{show:"",color:"light"}},[t._v(" Light Alert with "),r("a",{staticClass:"alert-link",attrs:{href:"#"}},[t._v("an example link")]),t._v(". ")]),r("CAlert",{attrs:{show:"",color:"dark"}},[t._v(" Dark Alert with "),r("CLink",{staticClass:"alert-link",attrs:{href:"#"}},[t._v("an example link")]),t._v(" . ")],1)],1)])],1)],1),r("CCol",{attrs:{col:"12",md:"6"}},[r("CCard",[r("CCardHeader",[r("CIcon",{attrs:{name:"cil-justify-center"}}),t._v(" Alerts "),r("small",[t._v("with additional content")])],1),r("CCardBody",[r("CAlert",{attrs:{show:"",color:"success"}},[r("h4",{staticClass:"alert-heading"},[t._v("Well done!")]),r("p",[t._v(" Aww yeah, you successfully read this important alert message. This example text is going to run a bit longer so that you can see how spacing within an alert works with this kind of content. ")]),r("hr"),r("p",{staticClass:"mb-0"},[t._v(" Whenever you need to, be sure to use margin utilities to keep things nice and tidy. ")])])],1)],1)],1),r("CCol",{attrs:{col:"12",md:"6"}},[r("CCard",[r("CCardHeader",[r("CIcon",{attrs:{name:"cil-justify-center"}}),t._v(" Alerts "),r("small",[t._v("dismissible")])],1),r("CCardBody",[r("CAlert",{attrs:{color:"secondary",closeButton:"",show:t.alert1},on:{"update:show":function(s){t.alert1=s}}},[t._v(" Dismissible Alert! ")]),r("CAlert",{staticClass:"alert-dismissible",attrs:{color:"secondary",show:t.alert2},on:{"update:show":function(s){t.alert2=s}}},[t._v(" Dismissible Alert with custom button! "),r("CButton",{staticClass:"position-absolute",staticStyle:{right:"10px",top:"50%",transform:"translateY(-50%)"},attrs:{color:"secondary"},on:{click:function(s){t.alert2=!1}}},[t._v(" Close ")])],1),r("CButton",{staticClass:"m-1",attrs:{color:"info"},on:{click:t.showDismissibleAlerts}},[t._v(" Show dismissible alerts ")])],1)],1),r("CCard",[r("CCardHeader",[r("CIcon",{attrs:{name:"cil-justify-center"}}),t._v(" Alerts "),r("small",[t._v("auto dismissible")])],1),r("CCardBody",[r("div",[r("CAlert",{attrs:{show:t.dismissCountDown,closeButton:"",color:"warning",fade:""},on:{"update:show":function(s){t.dismissCountDown=s}}},[t._v(" Alert will dismiss after "),r("strong",[t._v(t._s(t.dismissCountDown))]),t._v(" seconds... ")]),r("CAlert",{attrs:{show:t.dismissCountDown,closeButton:"",color:"info"},on:{"update:show":function(s){t.dismissCountDown=s}}},[t._v(" Alert will dismiss after "+t._s(t.dismissCountDown)+" seconds... "),r("CProgress",{attrs:{color:"info",max:t.dismissSecs,value:t.dismissCountDown,height:"4px"}})],1),r("CButton",{staticClass:"m-1",attrs:{color:"info"},on:{click:t.showAlert}},[t._v(" Show alert with timer ")])],1)])],1)],1)],1)},a=[],o={name:"Alerts",data:function(){return{dismissSecs:10,dismissCountDown:10,alert1:!0,alert2:!0}},methods:{countDownChanged:function(t){this.dismissCountDown=t},showAlert:function(){this.dismissCountDown=this.dismissSecs},showDismissibleAlerts:function(){var t=this;["alert1","alert2"].forEach((function(s){return t[s]=!0}))}}},l=o,i=r("2877"),n=Object(i["a"])(l,e,a,!1,null,null,null);s["default"]=n.exports}}]);
//# sourceMappingURL=chunk-2d226319.70ff2f2c.js.map