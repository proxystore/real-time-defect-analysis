'use strict';

/*
 * Call the API and update the length of the box
 */
function updateStatus(doc) {
    fetch("/api/status")
	.then(res => res.json())
	.then(
	(result) => {
	  doc.isLoaded = true,
	  doc.data = result,
	  doc.started = result.started
	},
	(error) => {
	  doc.isLoaded = true,
	  doc.data = result
	})
}


const StatBox = {
  data() {
	  return {
		  isLoaded: false,
		  error: null,
		  data: null,
		  started: false,
		  img_selection: 1
	  }
  },
  mounted() {
    updateStatus(this)
    fetch('/bokeh')
        .then(function(response) { return response.json() })
        .then(function(item) { return Bokeh.embed.embed_item(item) })
    setInterval(() => {updateStatus(this)}, 5000)
  }
}

// Create the application
const app = Vue.createApp(StatBox)

// Add a function to expand the range slider with time
app.directive('imgcount', (el, binding) => {
    el.max = binding.value
})
app.directive('imgchoice', (el, binding) => {
    const path = el.src.split("/")
    path[path.length - 1] = binding.value - 1  // Images are zero-indexed
    el.src = path.join("/")
})
app.mount('#status')
