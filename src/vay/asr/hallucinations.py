"""Hardcoded hallucination blacklists for Whisper ASR across supported languages."""

HALLUCINATION_BLACKLIST: dict[str, set[str]] = {
    "en": {
        "all right", "because i have not done that in the last few years", "but i grow them",
        "but i grow young", "bye", "bye bye", "can i go", "do you want me to turn it off",
        "have a good night guys", "ill see you next time", "i love you", "im happy to be here",
        "im just a form", "lets do that again", "nice", "no", "ok", "okay", "okay thank you",
        "so i am going to turn this off", "thanks for watching", "thank you", "thank you for having me",
        "thank you for listening", "thank you for today", "thank you for your time", "thank you so much",
        "thank you very much", "thank you very much for coming", "that i thought id just like to talk about for a minute",
        "yeah", "yes", "thank you so much for joining us", "thank you so much for joining us today",
        "thanks", "im going to be here to give you more money", "we have to be cool", "im so glad to be here",
        "i am so glad to have you all the time to work out with you", "thats where im at", "oh i got this job",
        "oh i got this stuff", "i feel ready to switch", "i say feel ready to switch try the switch to play",
        "im going to try the switch to side two", "im gonna try the switch to side too",
        "im going to try the switch to side 2", "i got this at the time", "oh whats that", "you know ryan",
        "thank you mr president", "thank you all", "aww", "aw", "and then what is the end of the family",
        "god bless you", "oh my god", "its awesome", "thats it", "thats the whole thing",
        "ill tell you what ill be saying", "im speaking i tell you what", "im speaking", "next slide",
        "were going to be a better place to go", "were going to be a better place", "all right thank you",
        "okay i got this stuff", "i said good already so im gonna switch",
        "im going to say its good already so im going to switch the side here",
        "i said good already so im gonna try to switch the bike", "i said good already so im going to switch the side too",
        "i got this at the time i dont know about it", "i dont know", "everything all right",
        "so were going to talk about this", "were going to talk about this", "i am very good",
        "and well see you next time", "im going to be a bad person", "were going to be a better person to go",
        "were going to be a better person", "right here", "well be right back",
        "oh i just said that i didnt see you in the hand", "i didnt see you in the hand", "i just went out",
        "i just went out and i just went out", "i just went out with his eyes", "uhhuh", "uhhuh thats where im at",
        "uhhuh thats where its actually", "thats where i should go", "its horrible", "i got this done",
        "youre doing good i got this stuff", "all right i got this", "im going to say its good already so im going to switch",
        "im going to switch the site", "i got this with the monolone", "im going to try the switch inside too",
        "you know i was like oh whats that", "you know right", "you know", "good", "all right were all alone",
        "all right you know all right"
    },
    "es": {
        "adiós", "adios", "chau", "gracias", "gracias por ver el vídeo", "gracias por ver el video",
        "gracias por ver este video", "muchas gracias", "muchas gracias por su participación",
        "muchas gracias por su participacion", "no", "no no no", "sí", "si", "sí sí sí sí", "si si si si",
        "suscríbete al canal", "suscribete al canal", "suscríbete al canal y dale a like",
        "suscribete al canal y dale a like", "y a las personas que nos ayudan", "y eso es todo por hoy",
        "y eso es todo por hoy nos vemos en el próximo video chau",
        "y eso es todo por hoy nos vemos en el proximo video chau",
        "y nos vemos en el próximo video", "y nos vemos en el proximo video",
        "muchas gracias y hasta pronto"
    },
    "ja": {
        "ご視聴ありがとうございました",
        "ご視聴ありがとうございます",
        "次の動画でお会いしましょう",
        "チャンネル登録をお願いします",
        "チャンネル登録よろしくお願いします",
        "高評価とチャンネル登録をお願いします",
        "また次回の動画でお会いしましょう",
        "最後までご視聴いただきありがとうございました"
    },
    "pt": {
        "e aí", "e ai", "é isso aí", "e isso ai", "é isso aí galera", "e isso ai galera",
        "então me lembrese me lembrese me lembrese", "entao me lembrese", "obrigado", "obrigada",
        "eu não sei o que é isso", "eu nao sei o que e isso", "eu não sei o que você está falando",
        "eu nao sei o que voce esta falando",
        "eu não sei o que você está falando mas eu não sei o que você está falando",
        "eu nao sei o que voce esta falando mas eu nao sei o que voce esta falando"
    },
    "ru": {
        "продолжение следует", "спасибо", "субтитры создавал dimatorzok", "субтитры сделал dimatorzok",
        "субтитры подогнал симон", "динамичная музыка", "и",
        "спасибо за субтитры алексею дубровскому", "субтитры делал dimatorzok"
    },
    "tr": {
        "abone olmayı unutmayın", "abone olmayi unutmayin", "abone olmayı unutmayınız",
        "abone olmayi unutmayiniz", "beni takip etmeyi unutmayın", "beni takip etmeyi unutmayin",
        "izlediğiniz için teşekkür ederim", "izlediginiz icin tesekkur ederim",
        "izlediğiniz için teşekkürler", "izlediginiz icin tesekkurler",
        "bir sonraki videoda görüşmek üzere", "bir sonraki videoda gorusmek uzere",
        "kanalıma abone olmayı unutmayın", "kanalima abone olmayi unutmayin"
    }
}
