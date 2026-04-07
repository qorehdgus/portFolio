package com.example.main;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.ResponseBody;


@Controller
public class mainController {
    @GetMapping("/hello")
    public String hello(Model model) {
        model.addAttribute("name", "Spring Boot + Thymeleaf");
        return "main"; // templates/hello.html 렌더링
    }

    @PostMapping("/path")
    @ResponseBody
    public String postMethodName(@RequestBody String entity) {
        
        return entity;
    }
    
}
