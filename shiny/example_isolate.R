library(shiny)

ui <- fluidPage(
  headerPanel("isolateのテスト"),
  sidebarPanel(
    sliderInput("obs1", "Number of observations1:",
                min = 0, max = 1000, value = 500),
    sliderInput("obs2", "Number of observations2:",
                min = 0, max = 1000, value = 500),
    actionButton("goButton", "Go Button", icon("refresh"))
    #actionLink("goLink", "Go Link", icon("refresh")),
    #submitButton("Update View", icon("refresh"))
  ),
  mainPanel(
    plotOutput("distPlot")
  )
)

server <- function(input, output, session) {
  output$distPlot <- renderPlot({
    # Take a dependency on input$goButton
    input$goButton
    
    # Use isolate() to avoid dependency on input$obs
    dist <- isolate(rnorm(input$obs1))
    hist(dist)
  })
}

shinyApp(ui, server)