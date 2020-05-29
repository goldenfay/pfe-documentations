<template>
  <CFooter :fixed="false">
    <div>
      <a href="#" target="_blank">MyUI</a>
      <span class="ml-1">&copy; {{new Date().getFullYear()}} USTHB.</span>
    </div>
    <div class="ml-auto">
      <span class="mr-1">Powered by</span>
      <a href="#" target="_blank">PFE MASTER 2020 CROWD COUNTING</a>
    </div>
  </CFooter>
</template>

<script>
export default {
  name: 'TheFooter'
}
</script>
