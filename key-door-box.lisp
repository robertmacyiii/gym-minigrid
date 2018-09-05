(clear-all)
(require-extra "blending")

(define-model test-blending
    (sgp :sim-hook "similarity_function"
         ;:blending-request-hook "new_blend_request"
         :tmp 1.0
         ;:seed (1 1) :bll nil :blc 5 :mp 1 :v t :blt t :esc t :ans .25 :rt -5)
         :seed (1 1) :bll nil :blc 5 :mp 1 :v t :blt t :esc t :ans nil :rt -5 :value->mag second)

  ;(chunk-type observation key wall fc door lockeddoor goal)
  (chunk-type observation key wall door  goal)
  ;(chunk-type decision key lockeddoor fc wall door goal get_key open_door goto_goal)
  (chunk-type decision key  wall door goal get_key open_door goto_goal)

  

  (p p1
     =imaginal>
       isa observation
       key =KV
       ;has_key =HK
       ;lockeddoor =DV
       wall =WV
       ;door_unlocked =DU
       door =UdV
       goal =GV
       ;fc =FC
     ?blending>
       state free
       buffer empty
       error nil
     ==>
     @imaginal>
     +blending>
       isa decision
       key =KV
       ;has_key =HK
       ;lockeddoor =DV
       wall =WV
       ;fc =FC
       door =UDv
       ;door_unlocked =DU
       goal =GV
      )

  
  (p p2
     =blending>
       isa decision
       get_key =GK
       open_door =GD
       goto_goal =GG
     ?blending>
       state free
     ==>
     ;!output! (blended value is =val)
     
     ; Overwrite the blended chunk to erase it and keep it 
     ; from being added to dm.  Not necessary, but keeps the 
     ; examples simpler.

         
     @blending>    
     !eval! ("record_response" 'get_key' =GK 'open_door' =GD 'goto_goal' =GG)
     ;;+blending>
     ;;  isa target
     ;;  key key-2)
     )
  
  )
